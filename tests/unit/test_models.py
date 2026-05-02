"""Tests for Pydantic document schemas in db.models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

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
# FridgeState
# ---------------------------------------------------------------------------


class TestFridgeState:
    """Tests for FridgeState schema validation."""

    def test_valid_fridge_state(self) -> None:
        """FridgeState accepts valid input with detected items."""
        state = FridgeState(
            user_id="u1",
            image_path="/img/fridge.jpg",
            detected_items=[
                DetectedItem(
                    name="apple",
                    bounding_box=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
                    confidence=0.95,
                )
            ],
            captured_at=datetime(2025, 1, 10, 8, 30),
        )
        assert state.user_id == "u1"
        assert len(state.detected_items) == 1
        assert state.detected_items[0].name == "apple"

    def test_fridge_state_defaults(self) -> None:
        """detected_items defaults to empty list; created_at auto-populates."""
        state = FridgeState(
            user_id="u1",
            image_path="/img/f.jpg",
            captured_at=datetime(2025, 1, 1),
        )
        assert state.detected_items == []
        assert isinstance(state.created_at, datetime)

    def test_fridge_state_missing_required(self) -> None:
        """FridgeState rejects missing required fields."""
        with pytest.raises(ValidationError):
            FridgeState()  # type: ignore[call-arg]

    def test_fridge_state_roundtrip(self) -> None:
        """Serialize → deserialize preserves data."""
        state = FridgeState(
            user_id="u1",
            image_path="/img/f.jpg",
            captured_at=datetime(2025, 1, 1),
        )
        data = state.model_dump(mode="json")
        restored = FridgeState.model_validate(data)
        assert restored.user_id == state.user_id


# ---------------------------------------------------------------------------
# MealLog
# ---------------------------------------------------------------------------


class TestMealLog:
    """Tests for MealLog schema validation."""

    def test_valid_meal_log(self) -> None:
        """MealLog accepts valid input."""
        log = MealLog(
            user_id="u1",
            meal_type=MealType.LUNCH,
            items=[
                MealItem(
                    name="chicken",
                    grams=200,
                    calories=330,
                    protein_g=62,
                    carbs_g=0,
                    fat_g=7,
                )
            ],
            logged_at=datetime(2025, 1, 10, 12, 30),
        )
        assert log.meal_type == MealType.LUNCH
        assert log.items[0].protein_g == 62

    def test_meal_type_string_coercion(self) -> None:
        """MealType accepts valid string values."""
        log = MealLog(
            user_id="u1",
            meal_type="dinner",  # type: ignore[arg-type]
            logged_at=datetime(2025, 1, 1),
        )
        assert log.meal_type == MealType.DINNER

    def test_invalid_meal_type(self) -> None:
        """MealLog rejects invalid meal types."""
        with pytest.raises(ValidationError):
            MealLog(
                user_id="u1",
                meal_type="brunch",  # type: ignore[arg-type]
                logged_at=datetime(2025, 1, 1),
            )

    def test_negative_grams_rejected(self) -> None:
        """MealItem rejects negative gram values."""
        with pytest.raises(ValidationError):
            MealItem(
                name="x",
                grams=-10,
                calories=0,
                protein_g=0,
                carbs_g=0,
                fat_g=0,
            )


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class TestUserProfile:
    """Tests for UserProfile schema validation."""

    def test_valid_user_profile(self) -> None:
        """UserProfile accepts valid input and applies defaults."""
        profile = UserProfile(user_id="u1", display_name="Rakshitha")
        assert profile.daily_targets.calories == 2000.0
        assert profile.dietary_restrictions == []

    def test_custom_macro_targets(self) -> None:
        """MacroTargets can be overridden."""
        targets = MacroTargets(calories=2500, protein_g=80, carbs_g=300, fat_g=90)
        profile = UserProfile(
            user_id="u1",
            display_name="Test",
            daily_targets=targets,
        )
        assert profile.daily_targets.protein_g == 80

    def test_user_profile_roundtrip(self) -> None:
        """Serialize → deserialize preserves data."""
        profile = UserProfile(
            user_id="u1",
            display_name="Test",
            dietary_restrictions=["vegan"],
        )
        data = profile.model_dump(mode="json")
        restored = UserProfile.model_validate(data)
        assert restored.dietary_restrictions == ["vegan"]

    def test_user_profile_missing_required(self) -> None:
        """UserProfile rejects missing required fields."""
        with pytest.raises(ValidationError):
            UserProfile()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Confidence bounds
# ---------------------------------------------------------------------------


class TestConfidenceBounds:
    """Test field constraints on DetectedItem."""

    def test_confidence_above_1_rejected(self) -> None:
        """Confidence > 1.0 is invalid."""
        with pytest.raises(ValidationError):
            DetectedItem(
                name="x",
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
                confidence=1.5,
            )

    def test_confidence_below_0_rejected(self) -> None:
        """Confidence < 0.0 is invalid."""
        with pytest.raises(ValidationError):
            DetectedItem(
                name="x",
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
                confidence=-0.1,
            )
