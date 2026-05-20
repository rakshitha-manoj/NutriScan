"""FastAPI application factory with async MongoDB lifespan."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import FastAPI

from api.routes import fridge_router, health_router, plan_router
from db.session import db_manager


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect to MongoDB on startup, disconnect on shutdown."""
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "nutriscan")
    await db_manager.connect(url=mongo_url, db_name=db_name)

    # Ensure uploads directory exists.
    Path("data/raw/uploads").mkdir(parents=True, exist_ok=True)

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
        version="0.1.0",
        lifespan=_lifespan,
    )

    application.include_router(health_router)
    application.include_router(fridge_router)
    application.include_router(plan_router)

    return application


# Uvicorn entry point: ``uvicorn api.main:app``
app = create_app()
