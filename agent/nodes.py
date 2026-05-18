"""Graph node functions for the meal-planning agent.

Each node is a pure synchronous function:
    ``def node_name(state: AgentState) -> dict[str, Any]``

Nodes read from *state*, call helpers in :mod:`agent.tools`, and return
a dict of state updates.  Exceptions are caught and written to
``state["error"]`` so the graph can continue gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.tools import compute_macro_deficit, load_recipes, score_recipe
from db.models import MacroTargets

if TYPE_CHECKING:
    from agent.state import AgentState

_MIN_PLAN_SIZE = 3
_SCORE_THRESHOLD = 0.2


def load_profile(state: AgentState) -> dict[str, Any]:
    """Validate that a user profile is present in state."""
    if state.get("user_profile") is None:
        return {"error": "user_profile not loaded"}
    return {}


def load_fridge(state: AgentState) -> dict[str, Any]:
    """Validate that a fridge state is present in state."""
    if state.get("fridge_state") is None:
        return {"error": "fridge_state not loaded"}
    return {}


def compute_deficit_node(state: AgentState) -> dict[str, Any]:
    """Compute remaining macro budget from today's meal logs."""
    profile = state.get("user_profile")
    if profile is None:
        return {"error": "user_profile required for deficit computation"}
    logs = list(state.get("meal_logs_today", []))
    deficit = compute_macro_deficit(profile.daily_targets, logs)
    return {"macro_deficit": deficit}


def filter_recipes(state: AgentState) -> dict[str, Any]:
    """Load and filter recipes by dietary restrictions and fridge overlap."""
    try:
        all_recipes = load_recipes()
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    profile = state.get("user_profile")
    fridge = state.get("fridge_state")

    restrictions = set()
    if profile is not None:
        restrictions = {r.lower() for r in profile.dietary_restrictions}

    # Filter by dietary restrictions: if user is "vegetarian",
    # keep recipes tagged "vegetarian" or "vegan".
    candidates: list[dict[str, Any]] = []
    for recipe in all_recipes:
        tags = {t.lower() for t in recipe.get("tags", [])}
        if restrictions and not restrictions.intersection(tags):
            continue
        candidates.append(recipe)

    # Soft-sort by fridge ingredient overlap (prefer recipes using
    # items the user actually has).
    fridge_items: set[str] = set()
    if fridge is not None:
        fridge_items = {item.name.lower() for item in fridge.detected_items}

    def _overlap(r: dict[str, Any]) -> int:
        uses = {n.lower() for n in r.get("uses_ingredients", [])}
        return len(uses.intersection(fridge_items))

    candidates.sort(key=_overlap, reverse=True)
    return {"candidate_recipes": candidates[:20]}


def score_recipes(state: AgentState) -> dict[str, Any]:
    """Score each candidate recipe against the macro deficit."""
    deficit = state.get("macro_deficit")
    fridge = state.get("fridge_state")
    candidates = state.get("candidate_recipes", [])

    if deficit is None or fridge is None:
        return {"error": "deficit and fridge required for scoring"}

    scored: list[dict[str, Any]] = []
    for recipe in candidates:
        s = score_recipe(recipe, deficit, fridge)
        scored.append({**recipe, "_score": round(s, 4)})

    scored.sort(key=lambda r: r["_score"], reverse=True)
    return {"scored_recipes": scored}


def select_plan(state: AgentState) -> dict[str, Any]:
    """Pick the top 3 recipes; fall back to full pool if needed."""
    scored = list(state.get("scored_recipes", []))
    deficit = state.get("macro_deficit")
    fridge = state.get("fridge_state")

    above = [r for r in scored if r.get("_score", 0) > _SCORE_THRESHOLD]

    if len(above) < _MIN_PLAN_SIZE and deficit is not None and fridge is not None:
        # Re-score from full recipe corpus without dietary filter.
        try:
            full = load_recipes()
        except FileNotFoundError:
            full = scored
        rescored = []
        for recipe in full:
            s = score_recipe(recipe, deficit, fridge)
            rescored.append({**recipe, "_score": round(s, 4)})
        rescored.sort(key=lambda r: r["_score"], reverse=True)
        above = rescored

    selected = above[:_MIN_PLAN_SIZE]

    # Compute projected macros.
    total_cal = sum(r.get("macros", {}).get("calories", 0) for r in selected)
    total_pro = sum(r.get("macros", {}).get("protein_g", 0) for r in selected)
    total_carb = sum(r.get("macros", {}).get("carbs_g", 0) for r in selected)
    total_fat = sum(r.get("macros", {}).get("fat_g", 0) for r in selected)

    projected = MacroTargets(
        calories=total_cal,
        protein_g=total_pro,
        carbs_g=total_carb,
        fat_g=total_fat,
    )
    return {"selected_plan": selected, "projected_macros": projected}


def persist_log(state: AgentState) -> dict[str, Any]:
    """Mark state for persistence (actual DB write in run_agent).

    Skips if an error occurred earlier in the pipeline.
    """
    if state.get("error"):
        return {}
    # Data is ready in selected_plan; run_agent handles async write.
    return {}
