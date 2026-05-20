"""Health check route — extracted from api/main.py."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse
from db.session import db_manager

router = APIRouter(tags=["meta"])

_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse(
        status="healthy",
        version=_VERSION,
        db="connected" if db_manager.is_connected else "disconnected",
    )
