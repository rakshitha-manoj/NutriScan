"""Pydantic v2 request / response schemas for the NutriScan API.

These are lean API-facing models — separate from the MongoDB document
schemas in ``db.models``.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic needs at runtime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    version: str
    db: str


class DetectedItemResponse(BaseModel):
    """Single item in a fridge analysis result."""

    label: str
    bounding_box: dict[str, float] = Field(..., description="{x, y, width, height}")
    detection_confidence: float
    estimated_grams: float
    uncertainty_grams: float
    freshness_score: float
    freshness_uncertainty: float
    freshness_label: str


class FridgeAnalyseResponse(BaseModel):
    """Response for ``POST /fridge/analyse``."""

    user_id: str
    captured_at: datetime
    items: list[DetectedItemResponse]
    pixel_to_cm2_ratio: float
    message: str


class MealPlanRequest(BaseModel):
    """Request body for ``POST /plan/daily``."""

    user_id: str
    meal_type: str = "dinner"


class ScoredRecipe(BaseModel):
    """A single recipe in the meal plan response."""

    id: str
    name: str
    score: float
    macros: dict[str, float] = Field(..., description="{calories, protein_g, carbs_g, fat_g}")
    tags: list[str]


class MealPlanResponse(BaseModel):
    """Response for ``POST /plan/daily``."""

    user_id: str
    meal_type: str
    selected_recipes: list[ScoredRecipe]
    projected_macros: dict[str, float]
    macro_deficit_before: dict[str, float]
    message: str
