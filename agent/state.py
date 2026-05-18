"""Agent state definition for the LangGraph meal-planning agent.

All state flows through this single TypedDict.  Nodes read from and
write to this structure; no side-channel data is permitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from db.models import FridgeState, MacroTargets, MealLog, UserProfile


class AgentState(TypedDict, total=False):
    """Typed state container for the meal-planning graph.

    Fields populated by ``run_agent`` before graph invocation:
        user_id, user_profile, fridge_state, meal_logs_today, meal_type.

    Fields populated by graph nodes during execution:
        macro_deficit, candidate_recipes, scored_recipes,
        selected_plan, projected_macros, error.
    """

    user_id: str
    user_profile: UserProfile | None
    fridge_state: FridgeState | None
    meal_logs_today: list[MealLog]
    macro_deficit: MacroTargets | None
    candidate_recipes: list[dict[str, Any]]
    scored_recipes: list[dict[str, Any]]
    selected_plan: list[dict[str, Any]]
    projected_macros: MacroTargets | None
    meal_type: str
    error: str | None
