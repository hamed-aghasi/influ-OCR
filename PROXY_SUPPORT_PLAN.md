# Adding System Proxy Support to Instagram Analyzer

## Summary

This document outlines how to add comprehensive proxy support (HTTP/HTTPS + SOCKS5) with username/password authentication, configured via environment variables.

## Current State

- **`requests`** (gemini_processor.py:136) - No proxy configured
- **`boto3`** (s3_storage.py:43) - No proxy configured
- No proxy-related code exists in the codebase

## Implementation Steps

### Step 1: Add Dependencies

**File:** `requirements.txt`

Add SOCKS5 support:
```
PySocks>=1.7.1
requests[socks]>=2.31.0
```

### Step 2: Update .env Template

**File:** `.env`

Add proxy environment variables:
```env
# Proxy Configuration (optional)
# HTTP/HTTPS Proxy: http://user:pass@host:port
HTTP_PROXY=
HTTPS_PROXY=

# SOCKS5 Proxy: socks5://user:pass@host:port
# SOCKS5_PROXY takes precedence over HTTP_PROXY if both are set
SOCKS5_PROXY=

# Set to bypass proxy for specific hosts
NO_PROXY=localhost,127.0.0.1
```

### Step 3: Create Proxy Utility Module

**File:** `processing/proxy_config.py` (new file)

```python
"""Proxy configuration utility for HTTP clients."""
import os
from typing import Optional, Dict

def get_proxy_config() -> Optional[Dict[str, str]]:
    """Get proxy configuration from environment variables.

    Priority: SOCKS5_PROXY > HTTPS_PROXY/HTTP_PROXY

    Returns dict for requests library or None if no proxy configured.
    """
    socks_proxy = os.getenv('SOCKS5_PROXY')
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY', http_proxy)

    if socks_proxy:
        return {
            'http': socks_proxy,
            'https': socks_proxy
        }
    elif http_proxy or https_proxy:
        return {
            'http': http_proxy,
            'https': https_proxy
        }
    return None


def get_boto3_proxy_config() -> Optional[Dict[str, str]]:
    """Get proxy config formatted for boto3/botocore."""
    return get_proxy_config()
```

### Step 4: Update Gemini Processor

**File:** `processing/gemini_processor.py`

Modify the `call_gemini_api()` function (around line 136):

```python
# Add import at top
from .proxy_config import get_proxy_config

# In call_gemini_api():
proxies = get_proxy_config()
response = requests.post(
    API_URL,
    headers=headers,
    json=data,
    timeout=60,
    proxies=proxies  # Add this parameter
)
```

### Step 5: Update S3 Storage

**File:** `processing/s3_storage.py`

Modify the `get_s3_client()` function (around line 30):

```python
# Add imports at top
from botocore.config import Config
from .proxy_config import get_boto3_proxy_config

def get_s3_client():
    """Get or create S3 client with optional proxy support."""
    global _s3_client

    if _s3_client is not None:
        return _s3_client

    if not is_s3_configured():
        logger.warning("S3 not configured - missing environment variables")
        return None

    try:
        import boto3

        # Configure proxy if available
        proxy_config = get_boto3_proxy_config()
        config = Config(proxies=proxy_config) if proxy_config else None

        _s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=config  # Add proxy config
        )
        logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None
```

### Step 6: Add Startup Logging (Optional)

**File:** `main.py`

Add logging at startup to show proxy status:

```python
from processing.proxy_config import get_proxy_config

# At app startup
proxy = get_proxy_config()
if proxy:
    logger.info(f"Proxy configured: HTTP={proxy.get('http')}, HTTPS={proxy.get('https')}")
else:
    logger.info("No proxy configured - using direct connection")
```

## Files Summary

| File | Action |
|------|--------|
| `requirements.txt` | Add PySocks dependency |
| `.env` | Add proxy variables |
| `processing/proxy_config.py` | **New** - Proxy utility |
| `processing/gemini_processor.py` | Add proxies param |
| `processing/s3_storage.py` | Add botocore Config |
| `main.py` | Log proxy status |

## Environment Variable Reference

| Variable | Format | Example |
|----------|--------|---------|
| `HTTP_PROXY` | `http://user:pass@host:port` | `http://admin:secret@proxy.corp.com:8080` |
| `HTTPS_PROXY` | `https://user:pass@host:port` | `https://admin:secret@proxy.corp.com:8080` |
| `SOCKS5_PROXY` | `socks5://user:pass@host:port` | `socks5://admin:secret@proxy.corp.com:1080` |
| `NO_PROXY` | Comma-separated hosts | `localhost,127.0.0.1,.internal.com` |

## Notes

- If no proxy env vars are set, app works as before (direct connection)
- SOCKS5 takes precedence if both `SOCKS5_PROXY` and `HTTP_PROXY` are set
- Credentials in URL are automatically handled by both `requests` and `boto3`
- `NO_PROXY` is respected by `requests` library automatically
- For SOCKS5 to work, `PySocks` must be installed
