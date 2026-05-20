"""LangGraph state machine for meal-planning.

``build_graph`` assembles the deterministic node pipeline.
``run_agent`` is the async entry point that pre-fetches all DB data,
invokes the synchronous graph, and persists results.
"""

from __future__ import annotations

import contextlib
from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import (
    compute_deficit_node,
    filter_recipes,
    load_fridge,
    load_profile,
    persist_log,
    score_recipes,
    select_plan,
)
from agent.state import AgentState
from agent.tools import (
    load_fridge_state,
    load_meal_logs_today,
    load_user_profile,
    persist_meal_log,
)


def build_graph() -> Any:
    """Build and compile the meal-planning StateGraph.

    Returns a compiled LangGraph that can be invoked with
    ``graph.invoke(initial_state)``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_profile", load_profile)
    graph.add_node("load_fridge", load_fridge)
    graph.add_node("compute_deficit", compute_deficit_node)
    graph.add_node("filter_recipes", filter_recipes)
    graph.add_node("score_recipes", score_recipes)
    graph.add_node("select_plan", select_plan)
    graph.add_node("persist_log", persist_log)

    graph.set_entry_point("load_profile")
    graph.add_edge("load_profile", "load_fridge")
    graph.add_edge("load_fridge", "compute_deficit")
    graph.add_edge("compute_deficit", "filter_recipes")
    graph.add_edge("filter_recipes", "score_recipes")
    graph.add_edge("score_recipes", "select_plan")
    graph.add_edge("select_plan", "persist_log")
    graph.add_edge("persist_log", END)

    return graph.compile()


async def run_agent(
    user_id: str,
    db: Any,
    meal_type: str = "dinner",
) -> dict[str, Any]:
    """Async entry point: pre-fetch DB data, run graph, persist results.

    Fetches the user's profile, latest fridge state, and today's meal
    logs from *db*, invokes the synchronous planning graph, and persists
    any selected recipes as meal logs. Returns the final agent state.
    """
    profile = None
    fridge = None
    logs: list[Any] = []
    error: str | None = None

    try:
        profile = await load_user_profile(user_id, db)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        error = f"Failed to load profile: {exc}"

    try:
        fridge = await load_fridge_state(user_id, db)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        error = error or f"Failed to load fridge: {exc}"

    try:
        logs = await load_meal_logs_today(user_id, db)
    except Exception as exc:  # noqa: BLE001
        logs = []
        error = error or f"Failed to load meal logs: {exc}"

    initial_state: dict[str, Any] = {
        "user_id": user_id,
        "user_profile": profile,
        "fridge_state": fridge,
        "meal_logs_today": logs,
        "macro_deficit": None,
        "candidate_recipes": [],
        "scored_recipes": [],
        "selected_plan": [],
        "projected_macros": None,
        "meal_type": meal_type,
        "error": error,
    }

    graph = build_graph()
    result: dict[str, Any] = graph.invoke(initial_state)

    if not result.get("error"):
        for recipe in result.get("selected_plan", []):
            with contextlib.suppress(Exception):
                await persist_meal_log(user_id, recipe, meal_type, db)

    return result
