"""Application configuration via pydantic-settings.

Phase 1 introduces this so secret_key + upload guards are enforced. More
settings move into here in later phases.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    secret_key: str = Field(..., min_length=16)
    admin_username: str = "admin"
    admin_password: str = "admin123"

    # Upload guards
    max_upload_bytes: int = 500 * 1024 * 1024
    max_zip_entries: int = 2_000
    max_zip_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024

    # ffmpeg
    ffmpeg_timeout_seconds: int = 600


settings = Settings()
