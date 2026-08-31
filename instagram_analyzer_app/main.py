"""FastAPI JSON API: upload, job dispatch, status, export. No UI.

Long-running ingest is enqueued to Celery; this process only handles HTTP.
"""

import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from processing.config import settings
from processing.db_client import (
    create_job,
    create_user,
    export_to_excel,
    get_all_jobs,
    get_job_by_id,
    get_user_count,
    set_job_task_id,
    update_job_status,
    verify_user,
)
from processing.errors import APIError, api_error_handler
from processing.filetypes import ALLOWED_EXTS, VIDEO_EXTS, ZIP_EXTS
from processing.logger import get_logger
from processing.naming import generate_job_id
from processing.s3_storage import download_json, get_file_url, is_s3_configured


logger = get_logger("main")

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processing_dir.mkdir(parents=True, exist_ok=True)


VIDEO_MAGIC = b"ftyp"  # appears at byte 4 in MP4 family
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _sniff_ok(header: bytes, ext: str) -> bool:
    if ext in ZIP_EXTS:
        return any(header.startswith(m) for m in ZIP_MAGIC)
    if ext in {".jpg", ".jpeg"}:
        return header.startswith(JPEG_MAGIC)
    if ext == ".png":
        return header.startswith(PNG_MAGIC)
    if ext in VIDEO_EXTS:
        return VIDEO_MAGIC in header[:32] or ext in {".mkv", ".avi", ".mov"}
    return False


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Under gunicorn every worker runs this; losers of the INSERT race get
    # create_user() == False (unique violation, logged) and that is fine.
    if get_user_count() == 0:
        if create_user(settings.admin_username, settings.admin_password):
            logger.info("Created default admin user: %s", settings.admin_username)
    yield


app = FastAPI(title="Instagram Analyzer", version="2.0.0", lifespan=_lifespan)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)
app.add_exception_handler(APIError, api_error_handler)


def _current_user(request: Request) -> Optional[str]:
    return request.session.get("user")


def _require_auth(request: Request) -> str:
    user = _current_user(request)
    if not user:
        raise APIError(401, "Not authenticated")
    return user


# ---------- Auth ----------


# Handlers below are deliberately sync (`def`, not `async def`): they call
# blocking code — bcrypt, psycopg2, openpyxl, boto3 — and as plain functions
# FastAPI runs them in its threadpool instead of stalling the event loop.


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if verify_user(username, password):
        request.session["user"] = username
        return {"status": "ok", "user": username}
    raise APIError(401, "Invalid credentials")


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


# ---------- Upload ----------


@app.post("/upload")
async def handle_upload(
    request: Request,
    file: UploadFile = File(...),
    campaign_date: str = Form(...),
    campaign_name: str = Form(...),
    product_name: str = Form(...),
    company: str = Form(...),
    user: str = Depends(_require_auth),
):
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTS:
        raise APIError(400, f"File extension {file_ext} not allowed")

    try:
        campaign_date_parsed = datetime.strptime(campaign_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise APIError(400, "Invalid date format (expected YYYY-MM-DD)") from exc

    header = await file.read(64)
    if not _sniff_ok(header, file_ext):
        raise APIError(400, "File contents do not match extension")
    await file.seek(0)

    job_id = generate_job_id(company, campaign_name)
    upload_path = settings.upload_dir / f"{job_id}{file_ext}"

    bytes_written = 0
    try:
        async with aiofiles.open(upload_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise APIError(
                        413,
                        f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)",
                    )
                await out.write(chunk)
    except APIError:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    if file_ext in ZIP_EXTS:
        try:
            _validate_zip(upload_path)
        except APIError:
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    if file_ext in VIDEO_EXTS:
        file_type = "video"
    elif file_ext in ZIP_EXTS:
        file_type = "zip"
    else:
        file_type = "image"

    if not create_job(job_id, campaign_date_parsed, campaign_name, product_name, company, file.filename, file_type):
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise APIError(500, "Could not create job record")

    from celery_app import celery  # local to avoid import-time worker dep

    try:
        result = celery.send_task("run_job", args=[job_id, str(upload_path), file_type])
    except Exception as exc:  # noqa: BLE001 — broker down must not leave a phantom queued job
        logger.error("Failed to enqueue job %s: %s", job_id, exc)
        update_job_status(job_id, "failed", "Could not enqueue job (queue unavailable)")
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise APIError(503, "Processing queue unavailable; try again shortly") from exc

    set_job_task_id(job_id, result.id)

    return {"status": "ok", "job_id": job_id, "task_id": result.id}


def _validate_zip(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise APIError(400, "Invalid ZIP file") from exc

    if len(entries) > settings.max_zip_entries:
        raise APIError(400, f"ZIP has too many entries ({len(entries)})")

    total_uncompressed = sum(e.file_size for e in entries)
    if total_uncompressed > settings.max_zip_uncompressed_bytes:
        raise APIError(
            400,
            f"ZIP would expand to {total_uncompressed} bytes (max {settings.max_zip_uncompressed_bytes})",
        )


# ---------- Job views ----------


@app.get("/jobs")
def jobs_list(
    status: Optional[str] = None,
    user: str = Depends(_require_auth),
):
    return {"jobs": get_all_jobs(limit=100, status_filter=status)}


@app.get("/export")
def export_excel(user: str = Depends(_require_auth)):
    excel_data = export_to_excel()
    if not excel_data:
        raise APIError(500, "Failed to generate Excel")
    filename = f"campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        iter([excel_data]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ---------- JSON API ----------


@app.get("/api/job/{job_id}")
def get_job_api(job_id: str, user: str = Depends(_require_auth)):
    job = get_job_by_id(job_id)
    if not job:
        raise APIError(404, "Job not found")
    return job


@app.get("/api/job/{job_id}/metrics")
def get_job_metrics(job_id: str, user: str = Depends(_require_auth)):
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")
    metrics = download_json(job_id)
    if not metrics:
        raise APIError(404, "Metrics not found")
    return JSONResponse(content=metrics)


@app.get("/api/job/{job_id}/metrics/download")
def download_job_metrics(job_id: str, user: str = Depends(_require_auth)):
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")
    url = get_file_url(job_id, "instagram_metrics.json", expires_in=3600)
    if not url:
        raise APIError(404, "Metrics not found")
    return RedirectResponse(url=url)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")
