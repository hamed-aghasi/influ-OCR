"""FastAPI app: upload form, job dispatch, status, export.

Long-running ingest is enqueued to Celery; this process only handles HTTP.
"""

import re
import secrets
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from processing.config import settings
from processing.db_client import (
    create_job,
    create_user,
    export_to_excel,
    get_all_jobs,
    get_job_by_id,
    get_user_count,
    verify_user,
)
from processing.errors import APIError, api_error_handler
from processing.logger import get_logger
from processing.s3_storage import download_json, get_file_url, is_s3_configured


logger = get_logger("main")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processing_dir.mkdir(parents=True, exist_ok=True)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ZIP_EXTS = {".zip"}
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS | ZIP_EXTS

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


app = FastAPI(title="Instagram Analyzer", version="2.0.0")
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)
app.add_exception_handler(APIError, api_error_handler)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _slug(text: str, limit: int) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace(" ", ""))[:limit]


def _generate_job_id(company: str, campaign_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = secrets.token_hex(2)  # 4 chars
    return f"{_slug(company, 15)}_{_slug(campaign_name, 15)}_{timestamp}_{suffix}"


def _current_user(request: Request) -> Optional[str]:
    return request.session.get("user")


def _require_auth(request: Request) -> str:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


@app.on_event("startup")
async def _startup() -> None:
    if get_user_count() == 0:
        if create_user(settings.admin_username, settings.admin_password):
            logger.info("Created default admin user: %s", settings.admin_username)


# ---------- Auth ----------


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if verify_user(username, password):
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"},
        status_code=401,
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ---------- Upload ----------


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request, user: str = Depends(_require_auth)):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "title": "Upload Campaign", "user": user},
    )


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

    job_id = _generate_job_id(company, campaign_name)
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

    create_job(job_id, campaign_date_parsed, campaign_name, product_name, company, file.filename, file_type)

    from celery_app import celery  # local to avoid import-time worker dep
    celery.send_task("run_job", args=[job_id, str(upload_path), file_type])

    return RedirectResponse(url=f"/status/{job_id}", status_code=303)


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


@app.get("/status/{job_id}", response_class=HTMLResponse)
async def status_page(request: Request, job_id: str, user: str = Depends(_require_auth)):
    job = get_job_by_id(job_id)
    if not job:
        raise APIError(404, "Job not found")
    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "title": "Job Status",
            "job": job,
            "auto_refresh": job.get("status") == "processing",
            "user": user,
        },
    )


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    status: Optional[str] = None,
    user: str = Depends(_require_auth),
):
    return templates.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "title": "All Jobs",
            "jobs": get_all_jobs(limit=100, status_filter=status),
            "status_filter": status,
            "user": user,
        },
    )


@app.get("/export")
async def export_excel(user: str = Depends(_require_auth)):
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
async def get_job_api(job_id: str, user: str = Depends(_require_auth)):
    job = get_job_by_id(job_id)
    if not job:
        raise APIError(404, "Job not found")
    return job


@app.get("/api/job/{job_id}/metrics")
async def get_job_metrics(job_id: str, user: str = Depends(_require_auth)):
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")
    metrics = download_json(job_id)
    if not metrics:
        raise APIError(404, "Metrics not found")
    return JSONResponse(content=metrics)


@app.get("/api/job/{job_id}/metrics/download")
async def download_job_metrics(job_id: str, user: str = Depends(_require_auth)):
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")
    url = get_file_url(job_id, "instagram_metrics.json", expires_in=3600)
    if not url:
        raise APIError(404, "Metrics not found")
    return RedirectResponse(url=url)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")
