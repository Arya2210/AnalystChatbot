import logging
import os
from functools import lru_cache
from typing import Optional

import dotenv
from pydantic import BaseModel, Field

dotenv.load_dotenv()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    temperature: float = 0.0  # Dropped default to 0.0 for deterministic SQL/Routing performance
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    timeout: int = 600


class GoogleGeminiSettings(LLMSettings):
    """Configuration matrix specifically for the Google Gemini GenAI suite."""
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    # Using 'gemini-1.5-pro' as default for robust multi-agent graph routing and SQL generation
    model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


class DatabaseSettings(BaseModel):
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    db_name: str = os.getenv("DB_NAME", "postgres")
    schema_name: str = os.getenv("DB_SCHEMA", "public")


class Settings(BaseModel):
    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
    # Core LLM Infrastructure
    gemini: GoogleGeminiSettings = Field(default_factory=GoogleGeminiSettings)
    
    # Runtime Guardrails
    allow_manipulation: bool = os.getenv("ALLOW_MANIPULATION", "False").lower() in ("true", "1", "t")
    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")


@lru_cache
def get_settings():
    settings = Settings()
    setup_logging()
    return settings