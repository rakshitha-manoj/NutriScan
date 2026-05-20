"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """NutriScan configuration.

    Values can be overridden via environment variables or a ``.env`` file.
    """

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "nutriscan"
    freshness_checkpoint_path: Path = Path("data/processed/freshness_best.pt")

    model_config = {"env_prefix": "NUTRISCAN_", "env_file": ".env"}


settings = Settings()
