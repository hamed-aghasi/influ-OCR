"""Job-id generation, shared by the upload route and the MinIO poller."""

import re
import secrets
from datetime import datetime


def slug(text: str, limit: int) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace(" ", ""))[:limit]


def generate_job_id(company: str, campaign_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = secrets.token_hex(2)  # 4 chars
    return f"{slug(company, 15)}_{slug(campaign_name, 15)}_{timestamp}_{suffix}"
