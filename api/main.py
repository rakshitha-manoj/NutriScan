"""FastAPI application factory with async MongoDB lifespan."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import FastAPI

from db.session import db_manager

_VERSION = "0.1.0"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect to MongoDB on startup, disconnect on shutdown."""
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "nutriscan")
    await db_manager.connect(url=mongo_url, db_name=db_name)
    try:
        yield
    finally:
        await db_manager.disconnect()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    application = FastAPI(
        title="NutriScan API",
        description=(
            "Agentic Nutrition Planner — fridge scanning, freshness estimation, and meal planning."
        ),
        version=_VERSION,
        lifespan=_lifespan,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Liveness / readiness probe."""
        return {
            "status": "healthy",
            "version": _VERSION,
            "db": "connected" if db_manager.is_connected else "disconnected",
        }

    return application


# Uvicorn entry point: ``uvicorn api.main:app``
app = create_app()
