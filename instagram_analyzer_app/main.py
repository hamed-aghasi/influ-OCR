"""
Instagram Analyzer - FastAPI Application

Main application with authentication and custom job IDs.
"""

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime, date
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
load_dotenv()

from processing.config import settings
from processing.errors import APIError, api_error_handler
from processing.logger import get_logger, job_logger
from processing import (
    create_job, update_job_status, save_job_metrics,
    get_all_jobs, get_job_by_id, export_to_excel,
    process_campaign_zip, classify_frames, extract_metrics_from_paths,
)
from processing.db_client import create_user, verify_user, get_user_count, init_database
from processing.s3_storage import download_json, get_file_url, is_s3_configured

logger = get_logger("main")

# Initialize FastAPI app
app = FastAPI(title="Instagram Analyzer", version="1.0.0")

# Session signing key is now required (>=16 chars) via settings.
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)
app.add_exception_handler(APIError, api_error_handler)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Directories
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', '/tmp/uploads'))
PROCESSING_DIR = Path(os.getenv('PROCESSING_DIR', '/tmp/processing'))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def generate_job_id(company: str, campaign_name: str) -> str:
    """Generate readable job ID from company and campaign name."""
    company_clean = re.sub(r'[^a-z0-9]', '', company.lower().replace(' ', ''))[:15]
    campaign_clean = re.sub(r'[^a-z0-9]', '', campaign_name.lower().replace(' ', ''))[:15]
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{company_clean}_{campaign_clean}_{timestamp}"


def get_current_user(request: Request) -> Optional[str]:
    """Get current logged-in user from session."""
    return request.session.get('user')


def require_auth(request: Request) -> str:
    """Dependency to require authentication."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


@app.on_event("startup")
async def startup_event():
    """Initialize database and create default admin user."""
    try:
        init_database()
        logger.info("Database initialized")

        # Create default admin if no users exist
        if get_user_count() == 0:
            admin_user = os.getenv('ADMIN_USERNAME', 'admin')
            admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123')
            create_user(admin_user, admin_pass)
            logger.info(f"Created default admin user: {admin_user}")
    except Exception as e:
        logger.error(f"Startup error: {e}")


# ============ Authentication Routes ============

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login form."""
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login."""
    if verify_user(username, password):
        request.session['user'] = username
        logger.info(f"User logged in: {username}")
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    user = request.session.get('user')
    request.session.clear()
    if user:
        logger.info(f"User logged out: {user}")
    return RedirectResponse(url="/login", status_code=303)


# ============ Protected Routes ============

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request, user: str = Depends(require_auth)):
    """Serve the upload form page."""
    return templates.TemplateResponse("upload.html", {"request": request, "title": "Upload Campaign", "user": user})


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
ZIP_EXTS = {'.zip'}
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS | ZIP_EXTS

ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _sniff_ok(header: bytes, ext: str) -> bool:
    if ext in ZIP_EXTS:
        return any(header.startswith(m) for m in ZIP_MAGIC)
    if ext in {'.jpg', '.jpeg'}:
        return header.startswith(JPEG_MAGIC)
    if ext == '.png':
        return header.startswith(PNG_MAGIC)
    if ext in VIDEO_EXTS:
        return b"ftyp" in header[:32] or ext in {'.mkv', '.avi', '.mov'}
    return False


def _validate_zip(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise APIError(400, "Invalid ZIP file") from exc
    if len(entries) > settings.max_zip_entries:
        raise APIError(400, f"ZIP has too many entries ({len(entries)})")
    total = sum(e.file_size for e in entries)
    if total > settings.max_zip_uncompressed_bytes:
        raise APIError(400, f"ZIP would expand to {total} bytes")


@app.post("/upload")
async def handle_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    campaign_date: str = Form(...),
    campaign_name: str = Form(...),
    product_name: str = Form(...),
    company: str = Form(...),
    user: str = Depends(require_auth)
):
    """Handle file upload and start background processing."""
    # Generate custom job ID
    job_id = generate_job_id(company, campaign_name)

    # Validate extension allowlist.
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTS:
        raise APIError(400, "File type not allowed")

    # Sniff magic bytes to ensure contents match extension.
    header = await file.read(64)
    if not _sniff_ok(header, file_ext):
        raise APIError(400, "File contents do not match extension")
    await file.seek(0)

    # Determine file type
    if file_ext in VIDEO_EXTS:
        file_type = 'video'
    elif file_ext in ZIP_EXTS:
        file_type = 'zip'
    else:
        file_type = 'image'

    # Parse campaign date
    try:
        campaign_date_parsed = datetime.strptime(campaign_date, '%Y-%m-%d').date()
    except ValueError:
        raise APIError(400, "Invalid date format (expected YYYY-MM-DD)")

    # Save uploaded file with size cap, async streaming.
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    bytes_written = 0
    try:
        async with aiofiles.open(upload_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    try:
                        upload_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise APIError(413, f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)")
                await out.write(chunk)
        logger.info(f"File saved: {upload_path}")
    except APIError:
        raise
    except OSError as e:
        logger.error(f"Failed to save file: {e}")
        raise APIError(500, "Failed to save file")

    if file_type == 'zip':
        try:
            _validate_zip(upload_path)
        except APIError:
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    # Create job
    create_job(job_id, campaign_date_parsed, campaign_name, product_name, company, file.filename, file_type)

    # Start background processing
    background_tasks.add_task(process_job, job_id, upload_path, file_type)

    return RedirectResponse(url=f"/status/{job_id}", status_code=303)


async def process_job(job_id: str, file_path: Path, file_type: str):
    """Background task to process uploaded file."""
    start_time = datetime.now()
    output_dir = PROCESSING_DIR / job_id

    try:
        logger.info(f"Starting job {job_id}")
        update_job_status(job_id, 'processing')

        # Step 1: Extract frames
        if file_type == 'zip':
            extraction_result = process_campaign_zip(file_path, PROCESSING_DIR, job_id)
            frame_paths = [Path(p) for p in extraction_result.get('frame_paths', [])]
            output_dir = Path(extraction_result.get('output_directory', output_dir))
        elif file_type == 'video':
            from processing.frame_extractor import extract_frames_from_video
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_count, frame_paths = extract_frames_from_video(file_path, output_dir / "frames")
        else:
            frame_paths = [file_path]

        total_frames = len(frame_paths)
        logger.info(f"Extracted {total_frames} frames")

        # Step 2: Classify frames (no longer copies to good/ + bad/ folders)
        good_paths, good_frames, bad_frames = [], [], []
        if frame_paths:
            result = classify_frames(frame_paths, output_dir=output_dir, job_id=job_id)
            good_paths = result.get('good_paths', [])
            good_frames = result.get('good_frames', [])
            bad_frames = result.get('bad_frames', [])
        logger.info(f"Classified: {len(good_frames)} good, {len(bad_frames)} bad")

        # Step 3: Extract metrics with OCR (perceptual-hash dedup happens inside)
        ocr_results = None
        if good_paths:
            ocr_results = extract_metrics_from_paths(good_paths, output_dir, job_id)
            logger.info(f"OCR completed: {ocr_results.get('unique_frames', 0)} unique frames")

        # Save metrics
        processing_time = int((datetime.now() - start_time).total_seconds())
        metrics = {
            'total_frames': total_frames,
            'good_frames': len(good_frames),
            'bad_frames': len(bad_frames),
            'processing_time_seconds': processing_time,
            'metrics_json': ocr_results if ocr_results else {}
        }
        save_job_metrics(job_id, metrics)
        update_job_status(job_id, 'completed')
        logger.info(f"Job {job_id} completed in {processing_time}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job_status(job_id, 'failed', str(e))
    finally:
        try:
            if file_path.exists():
                file_path.unlink()
        except:
            pass


@app.get("/status/{job_id}", response_class=HTMLResponse)
async def status_page(request: Request, job_id: str, user: str = Depends(require_auth)):
    """Show job status page."""
    job = get_job_by_id(job_id)
    if not job:
        raise APIError(404, "Job not found")
    return templates.TemplateResponse("status.html", {
        "request": request, "title": "Job Status", "job": job,
        "auto_refresh": job.get('status') == 'processing', "user": user
    })


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request, status: Optional[str] = None, user: str = Depends(require_auth)):
    """Show all jobs page."""
    jobs = get_all_jobs(limit=100, status_filter=status)
    return templates.TemplateResponse("jobs.html", {
        "request": request, "title": "All Jobs", "jobs": jobs,
        "status_filter": status, "user": user
    })


@app.get("/export")
async def export_excel(user: str = Depends(require_auth)):
    """Export all jobs to Excel."""
    excel_data = export_to_excel()
    if not excel_data:
        raise APIError(500, "Failed to generate Excel")
    filename = f"campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        iter([excel_data]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/health")
async def health_check():
    """Health check (no auth required)."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/job/{job_id}")
async def get_job_api(job_id: str, user: str = Depends(require_auth)):
    """API endpoint to get job details."""
    job = get_job_by_id(job_id)
    if not job:
        raise APIError(404, "Job not found")
    return job


@app.get("/api/job/{job_id}/metrics")
async def get_job_metrics(job_id: str, user: str = Depends(require_auth)):
    """Get JSON metrics for a job from S3."""
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")

    metrics = download_json(job_id)
    if not metrics:
        raise APIError(404, "Metrics not found")

    return JSONResponse(content=metrics)


@app.get("/api/job/{job_id}/metrics/download")
async def download_job_metrics(job_id: str, user: str = Depends(require_auth)):
    """Get a presigned URL to download metrics JSON from S3."""
    if not is_s3_configured():
        raise APIError(503, "S3 storage not configured")

    url = get_file_url(job_id, "instagram_metrics.json", expires_in=3600)
    if not url:
        raise APIError(404, "Metrics not found")

    return RedirectResponse(url=url)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
