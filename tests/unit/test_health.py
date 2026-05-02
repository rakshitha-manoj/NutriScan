"""Tests for the /health endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx


@pytest.mark.asyncio
async def test_health_returns_200(client: httpx.AsyncClient) -> None:
    """GET /health should return 200 with status, version, and db fields."""
    response = await client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert "version" in body
    assert "db" in body


@pytest.mark.asyncio
async def test_health_version_is_0_1_0(client: httpx.AsyncClient) -> None:
    """Version field should match the app version constant."""
    response = await client.get("/health")
    body = response.json()
    assert body["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_db_disconnected_without_mongo(client: httpx.AsyncClient) -> None:
    """Without a running MongoDB, db should report disconnected."""
    response = await client.get("/health")
    body = response.json()
    assert body["db"] == "disconnected"
