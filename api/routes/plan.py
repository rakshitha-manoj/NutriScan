"""Daily meal plan route."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from agent import run_agent
from api.dependencies import get_db
from api.schemas import MealPlanRequest, MealPlanResponse, ScoredRecipe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plan", tags=["plan"])

DbDep = Annotated[Any, Depends(get_db)]


@router.post("/daily", response_model=MealPlanResponse)
async def daily_plan(
    body: MealPlanRequest,
    db: DbDep,
) -> MealPlanResponse:
    """Generate a daily meal plan for a user."""
    try:
        result = await run_agent(user_id=body.user_id, db=db, meal_type=body.meal_type)
    except Exception:
        logger.exception("Agent error for user %s", body.user_id)
        raise HTTPException(status_code=500, detail="Internal server error.") from None

    error = result.get("error")
    if error is not None:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        raise HTTPException(status_code=422, detail=error)

    # Map selected plan to response schema.
    recipes = [
        ScoredRecipe(
            id=r.get("id", ""),
            name=r.get("name", ""),
            score=r.get("_score", 0.0),
            macros=r.get("macros", {}),
            tags=r.get("tags", []),
        )
        for r in result.get("selected_plan", [])
    ]

    projected = result.get("projected_macros")
    projected_dict: dict[str, float] = {}
    if projected is not None:
        projected_dict = {
            "calories": projected.calories,
            "protein_g": projected.protein_g,
            "carbs_g": projected.carbs_g,
            "fat_g": projected.fat_g,
        }

    deficit = result.get("macro_deficit")
    deficit_dict: dict[str, float] = {}
    if deficit is not None:
        deficit_dict = {
            "calories": deficit.calories,
            "protein_g": deficit.protein_g,
            "carbs_g": deficit.carbs_g,
            "fat_g": deficit.fat_g,
        }

    return MealPlanResponse(
        user_id=body.user_id,
        meal_type=body.meal_type,
        selected_recipes=recipes,
        projected_macros=projected_dict,
        macro_deficit_before=deficit_dict,
        message=f"Selected {len(recipes)} recipe(s) for {body.meal_type}.",
    )
