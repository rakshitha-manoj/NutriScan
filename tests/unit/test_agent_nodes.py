"""Unit tests for the LangGraph meal-planning agent.

All tests use in-memory state and mocked DB / recipe data.
No real MongoDB connection required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from agent.nodes import (
    compute_deficit_node,
    filter_recipes,
    load_fridge,
    load_profile,
    select_plan,
)
from agent.tools import compute_macro_deficit, score_recipe
from db.models import (
    BoundingBox,
    DetectedItem,
    FridgeState,
    MacroTargets,
    MealItem,
    MealLog,
    MealType,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=UTC)

_PROFILE = UserProfile(
    user_id="u1",
    display_name="Test",
    daily_targets=MacroTargets(calories=2000, protein_g=60, carbs_g=250, fat_g=70),
    dietary_restrictions=["vegetarian"],
)

_FRIDGE = FridgeState(
    user_id="u1",
    image_path="/fake.jpg",
    captured_at=_NOW,
    detected_items=[
        DetectedItem(
            name="apple",
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
            confidence=0.9,
            freshness_score=0.85,
        ),
        DetectedItem(
            name="banana",
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
            confidence=0.8,
            freshness_score=0.6,
        ),
    ],
)

_MEAL_LOG = MealLog(
    user_id="u1",
    meal_type=MealType.LUNCH,
    items=[
        MealItem(
            name="sandwich",
            grams=200,
            calories=500,
            protein_g=20,
            carbs_g=50,
            fat_g=20,
        )
    ],
    logged_at=_NOW,
)

_RECIPES = [
    {
        "id": "t1",
        "name": "Apple Oat Bowl",
        "ingredients": [{"name": "oats", "grams": 80}],
        "macros": {"calories": 380, "protein_g": 14, "carbs_g": 62, "fat_g": 8},
        "tags": ["vegetarian", "breakfast"],
        "uses_ingredients": ["apple"],
    },
    {
        "id": "t2",
        "name": "Steak Dinner",
        "ingredients": [{"name": "steak", "grams": 300}],
        "macros": {"calories": 700, "protein_g": 60, "carbs_g": 0, "fat_g": 40},
        "tags": ["high-protein", "dinner"],
        "uses_ingredients": [],
    },
    {
        "id": "t3",
        "name": "Banana Smoothie",
        "ingredients": [{"name": "banana", "grams": 120}],
        "macros": {"calories": 200, "protein_g": 5, "carbs_g": 40, "fat_g": 2},
        "tags": ["vegan", "snack"],
        "uses_ingredients": ["banana"],
    },
]


def _base_state(**overrides: object) -> dict[str, object]:
    """Build a minimal AgentState dict with optional overrides."""
    base: dict[str, object] = {
        "user_id": "u1",
        "user_profile": _PROFILE,
        "fridge_state": _FRIDGE,
        "meal_logs_today": [],
        "macro_deficit": None,
        "candidate_recipes": [],
        "scored_recipes": [],
        "selected_plan": [],
        "projected_macros": None,
        "meal_type": "dinner",
        "error": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# compute_macro_deficit tests
# ---------------------------------------------------------------------------


class TestComputeDeficit:
    """Tests for macro deficit computation."""

    def test_zero_deficit_when_logs_exceed_targets(self) -> None:
        """Each macro should floor at 0 when consumption exceeds target."""
        big_log = MealLog(
            user_id="u1",
            meal_type=MealType.LUNCH,
            items=[
                MealItem(
                    name="feast",
                    grams=1000,
                    calories=3000,
                    protein_g=100,
                    carbs_g=400,
                    fat_g=100,
                )
            ],
            logged_at=_NOW,
        )
        targets = MacroTargets(calories=2000, protein_g=60, carbs_g=250, fat_g=70)
        result = compute_macro_deficit(targets, [big_log])
        assert result.calories == 0.0
        assert result.protein_g == 0.0
        assert result.carbs_g == 0.0
        assert result.fat_g == 0.0

    def test_correct_remainder_with_partial_logs(self) -> None:
        """Deficit should equal targets minus consumed."""
        targets = MacroTargets(calories=2000, protein_g=60, carbs_g=250, fat_g=70)
        result = compute_macro_deficit(targets, [_MEAL_LOG])
        assert result.calories == pytest.approx(1500.0)
        assert result.protein_g == pytest.approx(40.0)
        assert result.carbs_g == pytest.approx(200.0)
        assert result.fat_g == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# score_recipe tests
# ---------------------------------------------------------------------------


class TestScoreRecipe:
    """Tests for recipe scoring logic."""

    def test_score_in_unit_interval(self) -> None:
        """Score must be between 0 and 1."""
        deficit = MacroTargets(calories=500, protein_g=20, carbs_g=60, fat_g=15)
        s = score_recipe(_RECIPES[0], deficit, _FRIDGE)
        assert 0.0 <= s <= 1.0

    def test_better_match_scores_higher(self) -> None:
        """A recipe closer to the deficit should score higher."""
        deficit = MacroTargets(calories=380, protein_g=14, carbs_g=62, fat_g=8)
        s_good = score_recipe(_RECIPES[0], deficit, _FRIDGE)
        s_bad = score_recipe(_RECIPES[1], deficit, _FRIDGE)
        assert s_good > s_bad


# ---------------------------------------------------------------------------
# Node-level tests
# ---------------------------------------------------------------------------


class TestNodes:
    """Tests for individual graph node functions."""

    @patch("agent.nodes.load_recipes", return_value=_RECIPES)
    def test_filter_restricts_by_diet(self, _mock: object) -> None:
        """Vegetarian filter should exclude non-vegetarian recipes."""
        state = _base_state()
        result = filter_recipes(state)  # type: ignore[arg-type]
        names = [r["name"] for r in result["candidate_recipes"]]
        # "Steak Dinner" has no vegetarian/vegan tag → excluded
        assert "Steak Dinner" not in names
        assert "Apple Oat Bowl" in names

    @patch("agent.nodes.load_recipes", return_value=_RECIPES)
    def test_select_plan_picks_top_3(self, _mock: object) -> None:
        """select_plan returns up to 3 recipes."""
        deficit = MacroTargets(calories=500, protein_g=20, carbs_g=60, fat_g=15)
        scored = [{**r, "_score": 0.5 + i * 0.1} for i, r in enumerate(_RECIPES)]
        state = _base_state(
            scored_recipes=scored,
            macro_deficit=deficit,
        )
        result = select_plan(state)  # type: ignore[arg-type]
        assert len(result["selected_plan"]) <= 3

    @patch("agent.nodes.load_recipes", return_value=_RECIPES)
    def test_select_plan_fallback(self, _mock: object) -> None:
        """Falls back to full pool when fewer than 3 score > 0.2."""
        deficit = MacroTargets(calories=500, protein_g=20, carbs_g=60, fat_g=15)
        scored = [{**_RECIPES[0], "_score": 0.1}]
        state = _base_state(
            scored_recipes=scored,
            macro_deficit=deficit,
        )
        result = select_plan(state)  # type: ignore[arg-type]
        # Should have re-scored from full corpus and returned results
        assert len(result["selected_plan"]) >= 1

    def test_load_profile_sets_error_when_missing(self) -> None:
        """Error field set when user_profile is None."""
        state = _base_state(user_profile=None)
        result = load_profile(state)  # type: ignore[arg-type]
        assert result.get("error") is not None

    def test_load_profile_passes_when_present(self) -> None:
        """No error when profile is present."""
        state = _base_state()
        result = load_profile(state)  # type: ignore[arg-type]
        assert result.get("error") is None

    def test_load_fridge_sets_error_when_missing(self) -> None:
        """Error field set when fridge_state is None."""
        state = _base_state(fridge_state=None)
        result = load_fridge(state)  # type: ignore[arg-type]
        assert result.get("error") is not None

    def test_compute_deficit_node_sets_deficit(self) -> None:
        """compute_deficit node populates macro_deficit."""
        state = _base_state(meal_logs_today=[_MEAL_LOG])
        result = compute_deficit_node(state)  # type: ignore[arg-type]
        assert result["macro_deficit"] is not None
        assert result["macro_deficit"].calories == pytest.approx(1500.0)

    @patch("agent.nodes.load_recipes", return_value=_RECIPES)
    def test_projected_macros_after_select(self, _mock: object) -> None:
        """projected_macros is not None after select_plan runs."""
        deficit = MacroTargets(calories=500, protein_g=20, carbs_g=60, fat_g=15)
        scored = [{**r, "_score": 0.8} for r in _RECIPES]
        state = _base_state(
            scored_recipes=scored,
            macro_deficit=deficit,
        )
        result = select_plan(state)  # type: ignore[arg-type]
        assert result["projected_macros"] is not None
        assert result["projected_macros"].calories > 0
