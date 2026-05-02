"""Shared test fixtures for NutriScan."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from api.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@pytest.fixture
def app() -> FastAPI:
    """Create a fresh FastAPI application for each test."""
    return create_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP client wired to the test app (no real server)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
