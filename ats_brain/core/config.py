"""Configuration settings for ATS Brain."""
import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    db_conn: Optional[str] = os.getenv("DB_CONN")
    GROK_API_KEY: str
    MODEL_NAME: str = "grok-4-fast-reasoning"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
