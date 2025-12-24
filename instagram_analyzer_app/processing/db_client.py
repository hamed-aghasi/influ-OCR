"""
Database Client Module

Handles PostgreSQL operations and Excel export.
Supports in-memory storage for local testing.
"""

import os
from datetime import datetime, date
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import io

from .logger import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
_memory_jobs: Dict[str, Dict] = {}
_memory_metrics: Dict[str, Dict] = {}
_memory_users: Dict[str, Dict] = {}


def is_database_available() -> bool:
    return DATABASE_URL is not None and DATABASE_URL.strip() != ""


def get_connection():
    if not is_database_available():
        return None
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def init_database():
    """Create tables if they don't exist."""
    if not is_database_available():
        logger.info("No database configured - using in-memory storage")
        return

    logger.info("Initializing database tables...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(100) PRIMARY KEY,
                    campaign_date DATE NOT NULL,
                    campaign_name VARCHAR(255) NOT NULL,
                    product_name VARCHAR(255) NOT NULL,
                    company VARCHAR(255) NOT NULL,
                    filename VARCHAR(255),
                    file_type VARCHAR(50),
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS job_metrics (
                    job_id VARCHAR(100) PRIMARY KEY REFERENCES jobs(id),
                    total_frames INTEGER,
                    good_frames INTEGER,
                    bad_frames INTEGER,
                    processing_time_seconds INTEGER,
                    metrics_json JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("Database tables initialized")


def create_job(job_id: str, campaign_date: date, campaign_name: str,
               product_name: str, company: str, filename: str, file_type: str) -> bool:
    """Create a new job record."""
    if not is_database_available():
        _memory_jobs[job_id] = {
            'id': job_id, 'campaign_date': campaign_date, 'campaign_name': campaign_name,
            'product_name': product_name, 'company': company, 'filename': filename,
            'file_type': file_type, 'status': 'processing', 'created_at': datetime.now(),
            'completed_at': None, 'error_message': None
        }
        logger.info(f"Created job {job_id} (in-memory)")
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO jobs (id, campaign_date, campaign_name, product_name, company, filename, file_type, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'processing')
                """, (job_id, campaign_date, campaign_name, product_name, company, filename, file_type))
                conn.commit()
                logger.info(f"Created job {job_id}")
                return True
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        return False


def update_job_status(job_id: str, status: str, error_message: str = None) -> bool:
    """Update job status."""
    if not is_database_available():
        if job_id in _memory_jobs:
            _memory_jobs[job_id]['status'] = status
            if status in ['completed', 'failed']:
                _memory_jobs[job_id]['completed_at'] = datetime.now()
            if error_message:
                _memory_jobs[job_id]['error_message'] = error_message
            return True
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if status in ['completed', 'failed']:
                    cur.execute("UPDATE jobs SET status=%s, completed_at=NOW(), error_message=%s WHERE id=%s",
                               (status, error_message, job_id))
                else:
                    cur.execute("UPDATE jobs SET status=%s WHERE id=%s", (status, job_id))
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
        return False


def save_job_metrics(job_id: str, metrics: dict) -> bool:
    """Save processing metrics for a job."""
    if not is_database_available():
        _memory_metrics[job_id] = {
            'job_id': job_id,
            'total_frames': metrics.get('total_frames', 0),
            'good_frames': metrics.get('good_frames', 0),
            'bad_frames': metrics.get('bad_frames', 0),
            'processing_time_seconds': metrics.get('processing_time_seconds'),
            'metrics_json': metrics.get('metrics_json', {}),
            'created_at': datetime.now()
        }
        logger.info(f"Saved metrics for job {job_id} (in-memory)")
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO job_metrics (job_id, total_frames, good_frames, bad_frames, processing_time_seconds, metrics_json)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (job_id) DO UPDATE SET
                        total_frames=EXCLUDED.total_frames, good_frames=EXCLUDED.good_frames,
                        bad_frames=EXCLUDED.bad_frames, processing_time_seconds=EXCLUDED.processing_time_seconds,
                        metrics_json=EXCLUDED.metrics_json
                """, (job_id, metrics.get('total_frames', 0), metrics.get('good_frames', 0),
                      metrics.get('bad_frames', 0), metrics.get('processing_time_seconds'),
                      json.dumps(metrics.get('metrics_json', {}))))
                conn.commit()
                logger.info(f"Saved metrics for job {job_id}")
                return True
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        return False


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by its ID with metrics."""
    if not is_database_available():
        if job_id in _memory_jobs:
            job = _memory_jobs[job_id].copy()
            if job_id in _memory_metrics:
                job.update(_memory_metrics[job_id])
            return job
        return None

    try:
        from psycopg2.extras import RealDictCursor
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT j.*, m.total_frames, m.good_frames, m.bad_frames, m.processing_time_seconds, m.metrics_json
                    FROM jobs j LEFT JOIN job_metrics m ON j.id = m.job_id WHERE j.id = %s
                """, (job_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get job: {e}")
        return None


def get_all_jobs(limit: int = 100, offset: int = 0, status_filter: str = None) -> List[Dict[str, Any]]:
    """Get all jobs with optional filtering."""
    if not is_database_available():
        jobs = []
        for job_id, job in _memory_jobs.items():
            if status_filter and job.get('status') != status_filter:
                continue
            job_copy = job.copy()
            if job_id in _memory_metrics:
                job_copy.update(_memory_metrics[job_id])
            jobs.append(job_copy)
        jobs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
        return jobs[offset:offset + limit]

    try:
        from psycopg2.extras import RealDictCursor
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT j.*, m.total_frames, m.good_frames, m.bad_frames, m.processing_time_seconds, m.metrics_json
                    FROM jobs j LEFT JOIN job_metrics m ON j.id = m.job_id
                """
                if status_filter:
                    query += " WHERE j.status = %s ORDER BY j.created_at DESC LIMIT %s OFFSET %s"
                    cur.execute(query, (status_filter, limit, offset))
                else:
                    query += " ORDER BY j.created_at DESC LIMIT %s OFFSET %s"
                    cur.execute(query, (limit, offset))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get jobs: {e}")
        return []


def export_to_excel(output_path: Optional[Path] = None) -> Optional[bytes]:
    """Export all jobs to Excel format."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

        jobs = get_all_jobs(limit=10000)
        if not jobs:
            logger.warning("No jobs to export")
            return None

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Campaign Results"

        headers = [
            "Job ID", "Campaign Date", "Campaign Name", "Product Name", "Company",
            "Filename", "File Type", "Status", "Created At", "Completed At",
            "Total Frames", "Good Frames", "Bad Frames", "Processing Time (s)",
            "Views (Max)", "Accounts Reached (Max)", "Followers (Max)", "Non-Followers (Max)",
            "Interactions (Max)", "Likes (Max)", "Replies (Max)", "Shares (Max)",
            "Profile Visits (Max)", "Follows (Max)", "Links Clicks (Max)", "Sticker Taps (Max)",
            "Navigation (Max)", "Forward (Max)", "Next Story (Max)", "Back (Max)",
            "Exited (Max)", "Profile Activity (Max)", "External Link Taps (Max)"
        ]

        # Styles
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        alt_fill = PatternFill(start_color="E9EBF5", end_color="E9EBF5", fill_type="solid")
        border = Border(left=Side(style='thin'), right=Side(style='thin'),
                       top=Side(style='thin'), bottom=Side(style='thin'))

        # Write headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font, cell.fill, cell.alignment, cell.border = header_font, header_fill, Alignment(horizontal='center'), border

        # Write data
        metric_keys = ['views', 'accounts_reached', 'followers', 'non_followers', 'interactions',
                      'likes', 'replies', 'shares', 'profile_visits', 'follows', 'links_clicks',
                      'sticker_taps', 'navigation', 'forward', 'next_story', 'back', 'exited',
                      'profile_activity', 'external_link_taps']

        for row_idx, job in enumerate(jobs, 2):
            metrics_json = job.get('metrics_json') or {}
            summary = metrics_json.get('summary', {}) if isinstance(metrics_json, dict) else {}

            row_data = [
                str(job.get('id', '')),
                job.get('campaign_date').isoformat() if job.get('campaign_date') else '',
                job.get('campaign_name', ''), job.get('product_name', ''), job.get('company', ''),
                job.get('filename', ''), job.get('file_type', ''), job.get('status', ''),
                job.get('created_at').isoformat() if job.get('created_at') else '',
                job.get('completed_at').isoformat() if job.get('completed_at') else '',
                job.get('total_frames', 0), job.get('good_frames', 0), job.get('bad_frames', 0),
                job.get('processing_time_seconds', 0)
            ] + [summary.get(k, {}).get('max', '') if summary else '' for k in metric_keys]

            for col, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col, value=value)
                cell.border = border
                if row_idx % 2 == 0:
                    cell.fill = alt_fill

        # Column widths
        for col, width in enumerate([36, 12, 20, 20, 20, 25, 10, 12, 20, 20, 12, 12, 12, 15] + [14]*19, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = width

        ws.freeze_panes = 'A2'

        if output_path:
            wb.save(output_path)
            return None

        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        logger.info("Excel file generated")
        return excel_buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to export Excel: {e}")
        return None


# ============ User Authentication Functions ============

def _hash_password(password: str) -> str:
    """Hash password using SHA256 with salt."""
    import hashlib
    salt = "instagram_analyzer_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def create_user(username: str, password: str) -> bool:
    """Create a new user."""
    password_hash = _hash_password(password)

    if not is_database_available():
        if username in _memory_users:
            logger.warning(f"User {username} already exists")
            return False
        _memory_users[username] = {
            'username': username,
            'password_hash': password_hash,
            'created_at': datetime.now()
        }
        logger.info(f"Created user {username} (in-memory)")
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    (username, password_hash)
                )
                conn.commit()
                logger.info(f"Created user {username}")
                return True
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return False


def verify_user(username: str, password: str) -> bool:
    """Verify user credentials."""
    password_hash = _hash_password(password)

    if not is_database_available():
        user = _memory_users.get(username)
        if user and user['password_hash'] == password_hash:
            return True
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT password_hash FROM users WHERE username = %s",
                    (username,)
                )
                result = cur.fetchone()
                if result and result[0] == password_hash:
                    return True
                return False
    except Exception as e:
        logger.error(f"Failed to verify user: {e}")
        return False


def get_user_count() -> int:
    """Get total number of users."""
    if not is_database_available():
        return len(_memory_users)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users")
                result = cur.fetchone()
                return result[0] if result else 0
    except Exception as e:
        logger.error(f"Failed to get user count: {e}")
        return 0
