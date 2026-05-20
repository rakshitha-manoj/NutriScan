"""Unit tests for API routes.

All dependencies are mocked — no real DB, no real model weights.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.main import create_app
from db.models import BoundingBox, MacroTargets

if TYPE_CHECKING:
    from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_portion_estimate() -> MagicMock:
    """Create a mock PortionEstimate."""
    est = MagicMock()
    est.label = "apple"
    est.bounding_box = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120)
    est.detection_confidence = 0.92
    est.estimated_grams = 180.0
    est.uncertainty_grams = 15.0
    est.pixel_to_cm2_ratio = 0.005
    return est


def _mock_freshness_prediction() -> MagicMock:
    """Create a mock FreshnessPrediction."""
    pred = MagicMock()
    pred.freshness_score = 0.85
    pred.uncertainty = 0.05
    pred.label = "fresh"
    return pred


def _mock_agent_state_ok() -> dict[str, Any]:
    return {
        "user_id": "u1",
        "error": None,
        "selected_plan": [
            {
                "id": "r01",
                "name": "Greek Yogurt Bowl",
                "_score": 0.82,
                "macros": {
                    "calories": 310,
                    "protein_g": 22,
                    "carbs_g": 48,
                    "fat_g": 5,
                },
                "tags": ["vegetarian", "breakfast"],
            }
        ],
        "projected_macros": MacroTargets(calories=310, protein_g=22, carbs_g=48, fat_g=5),
        "macro_deficit": MacroTargets(calories=1500, protein_g=40, carbs_g=200, fat_g=50),
    }


def _make_app_with_overrides(
    *,
    estimator: Any = None,
    freshness: Any = None,
    db: Any = None,
) -> FastAPI:
    """Create app with dependency overrides."""
    from api.dependencies import get_db, get_freshness_inference, get_portion_estimator

    application = create_app()
    if db is not None:
        application.dependency_overrides[get_db] = lambda: db
    if estimator is not None:
        application.dependency_overrides[get_portion_estimator] = lambda: estimator
    if freshness is not None:
        application.dependency_overrides[get_freshness_inference] = lambda: freshness
    return application


# ---------------------------------------------------------------------------
# /fridge/analyse tests
# ---------------------------------------------------------------------------


class TestFridgeAnalyse:
    """Tests for POST /fridge/analyse."""

    @pytest.mark.asyncio
    async def test_missing_file_returns_422(self) -> None:
        """No file uploaded → 422."""
        app = _make_app_with_overrides(
            estimator=MagicMock(),
            freshness=MagicMock(),
            db=MagicMock(),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/fridge/analyse", data={"user_id": "u1"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_content_type_returns_422(self) -> None:
        """Non-image file → 422."""
        app = _make_app_with_overrides(
            estimator=MagicMock(),
            freshness=MagicMock(),
            db=MagicMock(),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/fridge/analyse",
                data={"user_id": "u1"},
                files={"image": ("test.txt", b"hello", "text/plain")},
            )
        assert resp.status_code == 422
        assert "Unsupported file type" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_valid_jpeg_returns_200(self) -> None:
        """Valid JPEG → 200 with FridgeAnalyseResponse shape."""
        estimator = MagicMock()
        estimator.estimate.return_value = [_mock_portion_estimate()]

        freshness = MagicMock()
        freshness.predict.return_value = _mock_freshness_prediction()

        db = MagicMock()
        db.fridge_states = MagicMock()
        db.fridge_states.insert_one = AsyncMock()

        app = _make_app_with_overrides(estimator=estimator, freshness=freshness, db=db)
        transport = httpx.ASGITransport(app=app)

        # Create a minimal JPEG in memory.
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100), "red").save(buf, format="JPEG")
        buf.seek(0)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/fridge/analyse",
                data={"user_id": "u1"},
                files={"image": ("test.jpg", buf, "image/jpeg")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "u1"
        assert len(body["items"]) == 1
        assert "captured_at" in body

    @pytest.mark.asyncio
    async def test_items_length_matches_estimator(self) -> None:
        """Items list length matches mocked estimator output."""
        estimates = [_mock_portion_estimate(), _mock_portion_estimate()]
        estimator = MagicMock()
        estimator.estimate.return_value = estimates

        freshness = MagicMock()
        freshness.predict.return_value = _mock_freshness_prediction()

        db = MagicMock()
        db.fridge_states = MagicMock()
        db.fridge_states.insert_one = AsyncMock()

        app = _make_app_with_overrides(estimator=estimator, freshness=freshness, db=db)
        transport = httpx.ASGITransport(app=app)

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="JPEG")
        buf.seek(0)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/fridge/analyse",
                data={"user_id": "u1"},
                files={"image": ("test.jpg", buf, "image/jpeg")},
            )
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 2

    @pytest.mark.asyncio
    async def test_temp_file_deleted_after_processing(self) -> None:
        """Temp file must not exist after the request completes."""
        estimator = MagicMock()
        estimator.estimate.return_value = []

        db = MagicMock()
        db.fridge_states = MagicMock()
        db.fridge_states.insert_one = AsyncMock()

        app = _make_app_with_overrides(estimator=estimator, freshness=None, db=db)
        transport = httpx.ASGITransport(app=app)

        from pathlib import Path

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="JPEG")
        buf.seek(0)

        saved_paths: list[Path] = []
        original_write = Path.write_bytes

        def _tracking_write(self_path: Path, data: bytes) -> int:
            saved_paths.append(self_path)
            return original_write(self_path, data)

        with patch.object(Path, "write_bytes", _tracking_write):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                await c.post(
                    "/fridge/analyse",
                    data={"user_id": "u1"},
                    files={"image": ("test.jpg", buf, "image/jpeg")},
                )

        for p in saved_paths:
            assert not p.exists(), f"Temp file was not deleted: {p}"


# ---------------------------------------------------------------------------
# /plan/daily tests
# ---------------------------------------------------------------------------


class TestPlanDaily:
    """Tests for POST /plan/daily."""

    @pytest.mark.asyncio
    async def test_returns_200_on_success(self) -> None:
        """Valid request with mocked agent → 200."""
        db = MagicMock()
        app = _make_app_with_overrides(db=db)
        transport = httpx.ASGITransport(app=app)

        with patch("api.routes.plan.run_agent", new_callable=AsyncMock) as m:
            m.return_value = _mock_agent_state_ok()
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post(
                    "/plan/daily",
                    json={"user_id": "u1", "meal_type": "dinner"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["selected_recipes"]) == 1
        assert body["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_agent_error_returns_422(self) -> None:
        """Agent state with error → 422."""
        db = MagicMock()
        app = _make_app_with_overrides(db=db)
        transport = httpx.ASGITransport(app=app)

        state = _mock_agent_state_ok()
        state["error"] = "user_profile required for deficit computation"
        with patch("api.routes.plan.run_agent", new_callable=AsyncMock) as m:
            m.return_value = state
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post(
                    "/plan/daily",
                    json={"user_id": "u1"},
                )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self) -> None:
        """Error containing 'not found' → 404."""
        db = MagicMock()
        app = _make_app_with_overrides(db=db)
        transport = httpx.ASGITransport(app=app)

        state = _mock_agent_state_ok()
        state["error"] = "User not found for user_id='missing'"
        with patch("api.routes.plan.run_agent", new_callable=AsyncMock) as m:
            m.return_value = state
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post(
                    "/plan/daily",
                    json={"user_id": "missing"},
                )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_projected_macros_keys(self) -> None:
        """projected_macros must contain the expected keys."""
        db = MagicMock()
        app = _make_app_with_overrides(db=db)
        transport = httpx.ASGITransport(app=app)

        with patch("api.routes.plan.run_agent", new_callable=AsyncMock) as m:
            m.return_value = _mock_agent_state_ok()
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post(
                    "/plan/daily",
                    json={"user_id": "u1"},
                )
        macros = resp.json()["projected_macros"]
        assert set(macros.keys()) == {"calories", "protein_g", "carbs_g", "fat_g"}
