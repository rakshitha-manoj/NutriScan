"""Food category metadata for portion estimation.

Density and dimension values sourced from USDA FoodData Central
(fdc.nal.usda.gov) and standard produce reference tables.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoodMeta:
    """Metadata for a food category used in portion estimation.

    Stores the YOLO label, approximate density, typical height,
    expected aspect ratio, and weight of one typical unit.
    """

    yolo_label: str
    density_g_per_cm3: float
    typical_height_cm: float
    expected_aspect_ratio: float
    grams_per_unit: float


# Densities from USDA FoodData Central (fdc.nal.usda.gov).
# Heights and aspect ratios from standard produce sizing guides.
FOOD_CATEGORIES: dict[str, FoodMeta] = {
    "apple": FoodMeta(
        yolo_label="apple",
        density_g_per_cm3=0.85,  # USDA: ~0.8-0.9 g/cm3
        typical_height_cm=7.0,
        expected_aspect_ratio=1.0,  # roughly spherical
        grams_per_unit=182.0,  # USDA: 1 medium apple
    ),
    "banana": FoodMeta(
        yolo_label="banana",
        density_g_per_cm3=0.95,  # USDA: ~0.9-1.0 g/cm3
        typical_height_cm=3.5,
        expected_aspect_ratio=3.5,  # elongated
        grams_per_unit=118.0,  # USDA: 1 medium banana
    ),
    "orange": FoodMeta(
        yolo_label="orange",
        density_g_per_cm3=0.96,  # USDA: ~0.9-1.0 g/cm3
        typical_height_cm=7.5,
        expected_aspect_ratio=1.0,  # roughly spherical
        grams_per_unit=131.0,  # USDA: 1 medium orange
    ),
    "carrot": FoodMeta(
        yolo_label="carrot",
        density_g_per_cm3=1.04,
        typical_height_cm=3.0,
        expected_aspect_ratio=4.0,
        grams_per_unit=61.0,
    ),
    "broccoli": FoodMeta(
        yolo_label="broccoli",
        density_g_per_cm3=0.37,  # low density (florets)
        typical_height_cm=8.0,
        expected_aspect_ratio=1.2,
        grams_per_unit=148.0,
    ),
    "tomato": FoodMeta(
        yolo_label="tomato",  # not in COCO; maps to generic
        density_g_per_cm3=0.95,
        typical_height_cm=5.5,
        expected_aspect_ratio=1.1,
        grams_per_unit=123.0,
    ),
    "cucumber": FoodMeta(
        yolo_label="cucumber",  # not in COCO; maps to generic
        density_g_per_cm3=0.96,
        typical_height_cm=4.5,
        expected_aspect_ratio=3.0,
        grams_per_unit=301.0,
    ),
    "lemon": FoodMeta(
        yolo_label="lemon",  # not in COCO; maps to orange
        density_g_per_cm3=1.02,
        typical_height_cm=5.0,
        expected_aspect_ratio=1.2,
        grams_per_unit=58.0,
    ),
    "grape": FoodMeta(
        yolo_label="grape",  # not in COCO; maps to generic
        density_g_per_cm3=1.05,
        typical_height_cm=1.5,
        expected_aspect_ratio=1.0,
        grams_per_unit=5.0,  # single grape
    ),
    "strawberry": FoodMeta(
        yolo_label="strawberry",  # not in COCO; maps to generic
        density_g_per_cm3=0.96,
        typical_height_cm=3.0,
        expected_aspect_ratio=0.8,
        grams_per_unit=12.0,
    ),
}

# Reverse lookup: YOLO label string -> category key.
YOLO_LABEL_TO_CATEGORY: dict[str, str] = {
    meta.yolo_label: key for key, meta in FOOD_CATEGORIES.items()
}
