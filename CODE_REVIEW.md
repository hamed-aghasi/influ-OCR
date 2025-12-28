# Code Review: Instagram Analyzer

**Date:** 2024-12-24
**Reviewer:** Claude Code
**Overall Rating:** 6/10 (Functional MVP, needs hardening for production)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Issues](#critical-issues)
3. [Security Vulnerabilities](#security-vulnerabilities)
4. [Code Quality Issues](#code-quality-issues)
5. [Architecture Concerns](#architecture-concerns)
6. [Missing Features](#missing-features)
7. [File-by-File Analysis](#file-by-file-analysis)
8. [Recommended Action Plan](#recommended-action-plan)

---

## Executive Summary

The codebase is a functional FastAPI application for processing Instagram video campaigns. It demonstrates good module separation and basic error handling, but has several security vulnerabilities and code quality issues that should be addressed before production deployment.

### Strengths
- Clean module separation (processing/, templates/, static/)
- Type hints on most functions
- Docstrings present
- Environment-based configuration
- Consistent logging pattern
- Graceful fallbacks (in-memory DB, S3 optional)

### Weaknesses
- Weak password hashing
- Global mutable state
- Missing input validation
- No tests
- Bare except clauses
- No rate limiting

---

## Critical Issues

### 1. Weak Password Hashing

**File:** `processing/db_client.py:168-171`

```python
# CURRENT - INSECURE
def _hash_password(password: str) -> str:
    """Hash password using SHA256 with salt."""
    import hashlib
    salt = "instagram_analyzer_salt_2024"  # Hardcoded salt!
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
```

**Problem:**
- SHA256 is not designed for password hashing (too fast, vulnerable to brute force)
- Salt is hardcoded in source code (should be unique per password)
- No key stretching

**Fix:**

```python
# RECOMMENDED - SECURE
import bcrypt

def _hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())
```

**Required:** Add `bcrypt>=4.0.0` to requirements.txt

---

### 2. Global State Mutation

**Files:** `processing/frame_classifier.py:132`, `processing/frame_extractor.py:217`

```python
# CURRENT - PROBLEMATIC
def classify_frames(..., job_id: Optional[str] = None) -> Dict[str, Any]:
    global logger  # Mutating global state
    if job_id:
        logger = setup_logger(__name__, job_id)
```

**Problem:**
- Race conditions in concurrent requests
- Unpredictable behavior
- Hard to test

**Fix:**

```python
# RECOMMENDED - Pass logger as parameter
def classify_frames(
    frame_paths: List[Path],
    organize_files: bool = False,
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None  # Inject logger
) -> Dict[str, Any]:
    if logger is None:
        logger = setup_logger(__name__, job_id) if job_id else get_logger(__name__)

    # Use local logger variable throughout
```

---

## Security Vulnerabilities

### S1. Session Secret Key

**File:** `main.py:29`

```python
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "change-this-to-a-secure-random-key")
)
```

**Problem:** Default secret key in code. If SECRET_KEY env var is not set, sessions are predictable.

**Fix:**

```python
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    raise ValueError("SECRET_KEY environment variable must be set")

app.add_middleware(SessionMiddleware, secret_key=secret_key)
```

---

### S2. No Input Validation on File Uploads

**File:** `main.py:89-105`

```python
@app.post("/upload")
async def upload_file(...):
    # No validation on:
    # - File size limits
    # - File type verification (only checks extension)
    # - Filename sanitization
    # - ZIP bomb protection
```

**Fix:**

```python
import magic  # python-magic library

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_MIME_TYPES = ['application/zip', 'application/x-zip-compressed']

@app.post("/upload")
async def upload_file(file: UploadFile, ...):
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset

    if size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB")

    # Verify MIME type
    header = await file.read(2048)
    await file.seek(0)
    mime_type = magic.from_buffer(header, mime=True)

    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"Invalid file type: {mime_type}")

    # Continue processing...
```

---

### S3. No Rate Limiting

**Problem:** API endpoints have no rate limiting, vulnerable to:
- Brute force login attacks
- Resource exhaustion via repeated uploads

**Fix:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/login")
@limiter.limit("5/minute")  # 5 attempts per minute
async def login(request: Request, ...):
    ...

@app.post("/upload")
@limiter.limit("10/hour")  # 10 uploads per hour
async def upload_file(request: Request, ...):
    ...
```

**Required:** Add `slowapi>=0.1.9` to requirements.txt

---

### S4. SQL Injection Risk (Low)

**File:** `processing/db_client.py`

The code uses parameterized queries which is good, but some f-string formatting is used for table/column names:

```python
# Current - Generally safe but inconsistent
cursor.execute(f"SELECT * FROM {table} WHERE id = %s", (id,))
```

**Recommendation:** Use constants for table names, never interpolate user input into SQL structure.

---

## Code Quality Issues

### Q1. Bare Except Clauses

**Files:** Multiple locations

```python
# frame_classifier.py:209-210
except:
    pass

# frame_extractor.py:149-150
except:
    pass

# frame_classifier.py:190-191
except:
    pass
```

**Problem:** Silently swallows all exceptions including KeyboardInterrupt, SystemExit

**Fix:**

```python
# Catch specific exceptions
except (IOError, OSError) as e:
    logger.warning(f"Failed to copy file: {e}")

# Or at minimum, log the error
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

---

### Q2. Long Functions

**File:** `processing/db_client.py`

Several functions exceed 50 lines:
- `update_job_metrics()` - 80+ lines
- `create_job()` - 60+ lines

**Fix:** Extract helper functions

```python
# Before
def update_job_metrics(job_id: str, metrics: Dict) -> bool:
    # 80 lines of mixed logic

# After
def update_job_metrics(job_id: str, metrics: Dict) -> bool:
    if _use_in_memory():
        return _update_job_metrics_memory(job_id, metrics)
    return _update_job_metrics_postgres(job_id, metrics)

def _update_job_metrics_memory(job_id: str, metrics: Dict) -> bool:
    # Memory-specific logic

def _update_job_metrics_postgres(job_id: str, metrics: Dict) -> bool:
    # Postgres-specific logic
```

---

### Q3. Magic Numbers

**File:** `processing/frame_classifier.py:24-26`

```python
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.65
VERY_DARK_THRESHOLD = 80
```

**Problem:** These are module-level constants but should be configurable.

**Fix:**

```python
# config.py
from pydantic_settings import BaseSettings

class ClassifierConfig(BaseSettings):
    image_size: tuple = (224, 224)
    confidence_threshold: float = 0.65
    dark_threshold: int = 80

    class Config:
        env_prefix = "CLASSIFIER_"

classifier_config = ClassifierConfig()
```

---

### Q4. Inconsistent Error Responses

**File:** `main.py`

```python
# Sometimes returns dict
return {"error": "message"}

# Sometimes raises HTTPException
raise HTTPException(status_code=400, detail="message")

# Sometimes redirects
return RedirectResponse(url="/login", status_code=302)
```

**Fix:** Standardize error handling

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )
```

---

### Q5. No Type Hints on Some Functions

**File:** `processing/frame_classifier.py:29`

```python
def load_model():  # Missing return type
    """Load the classification model."""
```

**Fix:**

```python
from typing import Optional
from keras.layers import TFSMLayer

def load_model() -> Optional[TFSMLayer]:
    """Load the classification model."""
```

---

## Architecture Concerns

### A1. No Database Connection Pooling

**File:** `processing/db_client.py:25-35`

```python
def get_connection():
    """Get a database connection."""
    return psycopg2.connect(DATABASE_URL)
```

**Problem:** Creates new connection for every request. Inefficient and can exhaust connections.

**Fix:**

```python
from psycopg2 import pool

# Initialize pool once
connection_pool = pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    dsn=DATABASE_URL
)

def get_connection():
    return connection_pool.getconn()

def release_connection(conn):
    connection_pool.putconn(conn)

# Use as context manager
from contextlib import contextmanager

@contextmanager
def db_connection():
    conn = get_connection()
    try:
        yield conn
    finally:
        release_connection(conn)
```

---

### A2. Synchronous File Operations in Async Context

**File:** `main.py`

```python
@app.post("/upload")
async def upload_file(...):
    # Synchronous file operations block the event loop
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
```

**Fix:**

```python
import aiofiles

@app.post("/upload")
async def upload_file(...):
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
```

**Required:** Add `aiofiles>=23.0.0` to requirements.txt

---

### A3. No Request ID Tracking

**Problem:** Hard to trace requests through logs

**Fix:**

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Add to all log records
        logger = logging.getLogger()
        old_factory = logger.record_factory

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record

        logger.record_factory = record_factory
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
```

---

## Missing Features

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| Unit Tests | High | Medium | No tests exist. Need pytest + coverage |
| Integration Tests | High | Medium | Test API endpoints, DB operations |
| Health Check Improvements | Medium | Low | Add DB connectivity, S3 status |
| API Documentation | Medium | Low | OpenAPI docs need descriptions |
| Metrics/Monitoring | Medium | Medium | Prometheus metrics, request timing |
| Graceful Shutdown | Low | Low | Handle SIGTERM properly |
| Database Migrations | Medium | Medium | Use Alembic for schema changes |
| Caching | Low | Medium | Redis cache for repeated operations |

---

## File-by-File Analysis

### `main.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 29 | Default secret key | High | Require env var |
| 89-105 | No file validation | High | Add size/type checks |
| 147 | Sync file I/O | Medium | Use aiofiles |
| Various | Mixed error handling | Low | Standardize |

### `processing/db_client.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 168-171 | Weak password hashing | Critical | Use bcrypt |
| 25-35 | No connection pooling | Medium | Use psycopg2.pool |
| Various | Long functions | Low | Extract helpers |
| Various | Duplicated memory/postgres logic | Low | Use strategy pattern |

### `processing/frame_classifier.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 132 | Global logger mutation | Medium | Pass as parameter |
| 209-210 | Bare except | Medium | Catch specific |
| 29 | Missing return type | Low | Add type hint |

### `processing/frame_extractor.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 217 | Global logger mutation | Medium | Pass as parameter |
| 149-150 | Bare except | Medium | Catch specific |
| 36 | subprocess without timeout | Low | Add timeout |

### `processing/gemini_processor.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| Various | API key logged partially | Low | Remove from logs |
| Various | No retry logic | Medium | Add exponential backoff |

### `processing/s3_storage.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| 22 | Global client | Low | Use dependency injection |
| Various | No retry on failures | Medium | Add boto3 retry config |

### `processing/logger.py`

| Line | Issue | Severity | Fix |
|------|-------|----------|-----|
| - | No structured logging | Low | Use JSON formatter |
| - | No log level from env | Low | Make configurable |

---

## Recommended Action Plan

### Phase 1: Critical Security (Do Immediately)

1. **Replace password hashing with bcrypt**
   - File: `processing/db_client.py`
   - Add: `bcrypt>=4.0.0` to requirements.txt
   - Effort: 1 hour

2. **Require SECRET_KEY environment variable**
   - File: `main.py`
   - Effort: 15 minutes

3. **Add file upload validation**
   - File: `main.py`
   - Add: `python-magic>=0.4.27` to requirements.txt
   - Effort: 2 hours

### Phase 2: Code Quality (Next Sprint)

4. **Fix global logger mutations**
   - Files: `frame_classifier.py`, `frame_extractor.py`
   - Effort: 2 hours

5. **Replace bare except clauses**
   - Files: Multiple
   - Effort: 1 hour

6. **Add rate limiting**
   - File: `main.py`
   - Add: `slowapi>=0.1.9` to requirements.txt
   - Effort: 1 hour

### Phase 3: Architecture (Future)

7. **Add database connection pooling**
   - File: `processing/db_client.py`
   - Effort: 3 hours

8. **Add unit tests**
   - New: `tests/` directory
   - Add: `pytest`, `pytest-asyncio`, `pytest-cov`
   - Effort: 8+ hours

9. **Add database migrations**
   - Add: `alembic`
   - New: `migrations/` directory
   - Effort: 4 hours

---

## Updated Requirements.txt

```txt
# Current
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
jinja2>=3.1.2
psycopg2-binary>=2.9.9
opencv-python-headless>=4.8.0
numpy>=1.24.0
tensorflow>=2.15.0
keras>=3.0.0
requests>=2.31.0
itsdangerous>=2.1.0
boto3>=1.34.0

# Add for security
bcrypt>=4.0.0
python-magic>=0.4.27
slowapi>=0.1.9

# Add for quality
aiofiles>=23.0.0

# Add for testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0  # For testing FastAPI
```

---

## Conclusion

The codebase is a solid MVP but requires security hardening before production deployment. The most critical issue is the weak password hashing which should be fixed immediately. The code quality issues, while not critical, will make the codebase harder to maintain as it grows.

**Recommended Priority:**
1. Fix password hashing (Critical)
2. Add file validation (High)
3. Fix global state mutations (Medium)
4. Add tests (Medium)
5. Add connection pooling (Low)
