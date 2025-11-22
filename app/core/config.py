"""
Configuration module for BOT GPT backend.
Loads and validates settings from environment variables.
"""
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class Settings:
    """
    Application settings loaded from environment variables.
    
    Validates critical configuration on initialization.
    """

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./bot_gpt.db"
    )

    # LLM Configuration
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE_URL: str = os.getenv(
        "LLM_API_BASE_URL", "https://api.groq.com/openai"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # RAG Configuration
    RAG_MODEL: str = os.getenv(
        "RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RAG_INDEX_DIR: str = os.getenv("RAG_INDEX_DIR", "./data/indexes")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    # Conversation Configuration
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # API Configuration
    API_TITLE: str = "BOT GPT Backend"
    API_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    def __init__(self) -> None:
        """Validate critical settings on initialization."""
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate critical configuration values."""
        if self.LLM_TIMEOUT <= 0:
            raise ValueError("LLM_TIMEOUT must be positive")
        if self.RAG_TOP_K <= 0:
            raise ValueError("RAG_TOP_K must be positive")
        if self.MAX_HISTORY_MESSAGES <= 0:
            raise ValueError("MAX_HISTORY_MESSAGES must be positive")
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

    @property
    def llm_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return bool(self.LLM_API_KEY and self.LLM_API_BASE_URL)


def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Singleton settings instance.
        
    Raises:
        ValueError: If critical settings are invalid.
    """
    return Settings()
