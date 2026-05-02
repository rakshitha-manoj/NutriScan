"""MongoDB document schemas for NutriScan.

Three Pydantic v2 models representing the core domain:
- FridgeState: snapshot of detected items from a fridge photo
- MealLog: a single meal with nutritional breakdown
- UserProfile: user preferences and daily macro targets
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Embedded sub-models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x_min: float = Field(..., description="Left edge (px)")
    y_min: float = Field(..., description="Top edge (px)")
    x_max: float = Field(..., description="Right edge (px)")
    y_max: float = Field(..., description="Bottom edge (px)")


class DetectedItem(BaseModel):
    """A single food item detected in a fridge image."""

    name: str = Field(..., description="Predicted food label")
    bounding_box: BoundingBox = Field(..., description="Location in image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    freshness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Predicted freshness (1.0 = freshest)",
    )
    estimated_grams: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated portion weight in grams",
    )
    expiry_date: datetime | None = Field(
        default=None,
        description="Estimated expiry timestamp",
    )


class MealType(StrEnum):
    """Supported meal categories."""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class MealItem(BaseModel):
    """Nutritional data for a single food item in a meal."""

    name: str = Field(..., description="Food item name")
    grams: float = Field(..., ge=0.0, description="Weight in grams")
    calories: float = Field(..., ge=0.0, description="Energy (kcal)")
    protein_g: float = Field(..., ge=0.0, description="Protein (g)")
    carbs_g: float = Field(..., ge=0.0, description="Carbohydrates (g)")
    fat_g: float = Field(..., ge=0.0, description="Fat (g)")


class MacroTargets(BaseModel):
    """Daily macronutrient targets for a user."""

    calories: float = Field(default=2000.0, ge=0.0, description="Daily calorie target (kcal)")
    protein_g: float = Field(default=50.0, ge=0.0, description="Daily protein target (g)")
    carbs_g: float = Field(default=250.0, ge=0.0, description="Daily carbohydrate target (g)")
    fat_g: float = Field(default=70.0, ge=0.0, description="Daily fat target (g)")


# ---------------------------------------------------------------------------
# Top-level document models
# ---------------------------------------------------------------------------


class FridgeState(BaseModel):
    """Snapshot of a fridge scan — one image, many detected items.

    Maps to the ``fridge_states`` MongoDB collection.
    """

    user_id: str = Field(..., description="Owning user identifier")
    image_path: str = Field(..., description="Path or URL to the original image")
    detected_items: list[DetectedItem] = Field(
        default_factory=list,
        description="Items detected in the image",
    )
    captured_at: datetime = Field(..., description="When the photo was taken")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "image_path": "/data/raw/fridge_001.jpg",
                    "detected_items": [
                        {
                            "name": "apple",
                            "bounding_box": {
                                "x_min": 10,
                                "y_min": 20,
                                "x_max": 110,
                                "y_max": 120,
                            },
                            "confidence": 0.95,
                            "freshness_score": 0.85,
                            "estimated_grams": 180.0,
                            "expiry_date": "2025-01-15T00:00:00",
                        }
                    ],
                    "captured_at": "2025-01-10T08:30:00",
                }
            ]
        }
    }


class MealLog(BaseModel):
    """A single logged meal with nutritional breakdown.

    Maps to the ``meal_logs`` MongoDB collection.
    """

    user_id: str = Field(..., description="Owning user identifier")
    meal_type: MealType = Field(..., description="Category of the meal")
    items: list[MealItem] = Field(
        default_factory=list,
        description="Food items consumed in this meal",
    )
    logged_at: datetime = Field(..., description="When the meal was consumed")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "meal_type": "lunch",
                    "items": [
                        {
                            "name": "grilled chicken",
                            "grams": 200.0,
                            "calories": 330.0,
                            "protein_g": 62.0,
                            "carbs_g": 0.0,
                            "fat_g": 7.0,
                        }
                    ],
                    "logged_at": "2025-01-10T12:30:00",
                }
            ]
        }
    }


class UserProfile(BaseModel):
    """User preferences, dietary restrictions, and macro targets.

    Maps to the ``user_profiles`` MongoDB collection.
    """

    user_id: str = Field(..., description="Unique user identifier")
    display_name: str = Field(..., description="User display name")
    daily_targets: MacroTargets = Field(
        default_factory=lambda: MacroTargets(),
        description="Daily macronutrient goals",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="E.g. ['vegetarian', 'gluten-free']",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Profile creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last profile update timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "display_name": "Rakshitha",
                    "daily_targets": {
                        "calories": 2200.0,
                        "protein_g": 60.0,
                        "carbs_g": 275.0,
                        "fat_g": 73.0,
                    },
                    "dietary_restrictions": ["vegetarian"],
                }
            ]
        }
    }
