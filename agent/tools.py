"""Standalone helper functions used by agent graph nodes.

Async functions handle MongoDB access; synchronous functions handle
computation (macro arithmetic, recipe scoring).  All data access
for agent nodes flows through these helpers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from db.models import FridgeState, MacroTargets, MealLog, MealType, UserProfile

# ---------------------------------------------------------------------------
# Async MongoDB helpers (called by run_agent, not by sync nodes)
# ---------------------------------------------------------------------------

_RECIPES_PATH = Path(__file__).resolve().parent.parent / "data" / "recipes.json"


async def load_user_profile(user_id: str, db: Any) -> UserProfile:
    """Fetch a :class:`UserProfile` from MongoDB.

    Raises:
        ValueError: If no profile exists for *user_id*.
    """
    doc = await db.user_profiles.find_one({"user_id": user_id})
    if doc is None:
        msg = f"No user profile found for user_id={user_id!r}"
        raise ValueError(msg)
    doc.pop("_id", None)
    return UserProfile(**doc)


async def load_fridge_state(user_id: str, db: Any) -> FridgeState:
    """Fetch the most recent :class:`FridgeState` for a user.

    Raises:
        ValueError: If no fridge state exists for *user_id*.
    """
    cursor = db.fridge_states.find({"user_id": user_id}).sort("captured_at", -1)
    doc = await cursor.to_list(length=1)
    if not doc:
        msg = f"No fridge state found for user_id={user_id!r}"
        raise ValueError(msg)
    doc[0].pop("_id", None)
    return FridgeState(**doc[0])


async def load_meal_logs_today(user_id: str, db: Any) -> list[MealLog]:
    """Fetch today's meal logs for a user."""
    today_start = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    cursor = db.meal_logs.find({"user_id": user_id, "logged_at": {"$gte": today_start}})
    docs = await cursor.to_list(length=100)
    logs: list[MealLog] = []
    for d in docs:
        d.pop("_id", None)
        logs.append(MealLog(**d))
    return logs


async def persist_meal_log(
    user_id: str,
    recipe: dict[str, Any],
    meal_type: str,
    db: Any,
) -> None:
    """Write a single MealLog document to MongoDB."""
    from db.models import MealItem

    items = [
        MealItem(
            name=recipe["name"],
            grams=sum(i["grams"] for i in recipe.get("ingredients", [])),
            calories=recipe["macros"]["calories"],
            protein_g=recipe["macros"]["protein_g"],
            carbs_g=recipe["macros"]["carbs_g"],
            fat_g=recipe["macros"]["fat_g"],
        )
    ]
    log = MealLog(
        user_id=user_id,
        meal_type=MealType(meal_type),
        items=items,
        logged_at=datetime.now(tz=UTC),
    )
    await db.meal_logs.insert_one(log.model_dump())


# ---------------------------------------------------------------------------
# Synchronous computation helpers (called directly by graph nodes)
# ---------------------------------------------------------------------------


def load_recipes(path: Path | None = None) -> list[dict[str, Any]]:
    """Read and validate the local recipe corpus.

    Raises:
        FileNotFoundError: If the recipes file does not exist.
    """
    rpath = path or _RECIPES_PATH
    if not rpath.exists():
        msg = f"Recipe file not found: {rpath}"
        raise FileNotFoundError(msg)
    with rpath.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = "recipes.json must contain a JSON array"
        raise ValueError(msg)
    return data


def compute_macro_deficit(
    targets: MacroTargets,
    logs: list[MealLog],
) -> MacroTargets:
    """Compute remaining macro budget after today's consumed meals."""
    consumed_cal = 0.0
    consumed_pro = 0.0
    consumed_carb = 0.0
    consumed_fat = 0.0
    for log in logs:
        for item in log.items:
            consumed_cal += item.calories
            consumed_pro += item.protein_g
            consumed_carb += item.carbs_g
            consumed_fat += item.fat_g
    return MacroTargets(
        calories=max(0.0, targets.calories - consumed_cal),
        protein_g=max(0.0, targets.protein_g - consumed_pro),
        carbs_g=max(0.0, targets.carbs_g - consumed_carb),
        fat_g=max(0.0, targets.fat_g - consumed_fat),
    )


def score_recipe(
    recipe: dict[str, Any],
    deficit: MacroTargets,
    fridge: FridgeState,
) -> float:
    """Score a recipe against the macro deficit and fridge contents.

    Returns:
        A score in [0, 1] combining macro fit (70%) and freshness
        bonus (30%).
    """
    macros = recipe.get("macros", {})
    # Mean absolute relative error for each macro vs deficit.
    pairs = [
        (macros.get("calories", 0), deficit.calories),
        (macros.get("protein_g", 0), deficit.protein_g),
        (macros.get("carbs_g", 0), deficit.carbs_g),
        (macros.get("fat_g", 0), deficit.fat_g),
    ]
    errors: list[float] = []
    for recipe_val, target_val in pairs:
        denom = max(target_val, 1.0)
        errors.append(abs(recipe_val - target_val) / denom)
    macro_fit = max(0.0, min(1.0, 1.0 - sum(errors) / len(errors)))

    # Freshness bonus: mean freshness of fridge items in recipe.
    uses = {name.lower() for name in recipe.get("uses_ingredients", [])}
    fridge_names = {
        item.name.lower(): item.freshness_score
        for item in fridge.detected_items
        if item.freshness_score is not None
    }
    matched = [fridge_names[n] for n in uses if n in fridge_names]
    freshness_bonus = sum(matched) / len(matched) if matched else 0.0

    return 0.7 * macro_fit + 0.3 * freshness_bonus
