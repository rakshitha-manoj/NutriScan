"""Reference object detection and pixel-to-cm2 ratio estimation.

Uses a known reference object (plate, bowl, cutting board) to calibrate
the mapping from pixel area to real-world area in cm2.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.models import BoundingBox

# Reference plate dimensions (standard dinner plate).
PLATE_DIAMETER_CM: float = 26.0
PLATE_AREA_CM2: float = math.pi * (PLATE_DIAMETER_CM / 2) ** 2

# YOLO/COCO labels that can serve as reference objects.
REFERENCE_LABELS: frozenset[str] = frozenset(
    {
        "bowl",
        "dining table",
        "cup",
    }
)

# Fallback assumption: image captures ~60cm x 45cm of real surface.
_FALLBACK_REAL_WIDTH_CM: float = 60.0
_FALLBACK_REAL_HEIGHT_CM: float = 45.0


def detect_reference_object(
    boxes: list[BoundingBox],
    labels: list[str],
) -> float | None:
    """Find a known reference object and return its pixel area.

    Scans *boxes* and *labels* for a label matching
    :data:`REFERENCE_LABELS`. Returns the pixel area of the first
    matched box, or ``None`` if no reference object is detected.
    """
    for bbox, label in zip(boxes, labels, strict=True):
        if label.lower() in REFERENCE_LABELS:
            w = bbox.x_max - bbox.x_min
            h = bbox.y_max - bbox.y_min
            return float(w * h)
    return None


def estimate_pixel_to_cm2_ratio(
    reference_pixel_area: float | None,
    image_width: int,
    image_height: int,
) -> float:
    """Estimate the cm2-per-pixel2 conversion ratio.

    If a reference object was detected, uses its known real-world area
    (plate). Otherwise, falls back to a heuristic assuming the image
    captures roughly 60cm x 45cm of real surface.

    Returns:
        Ratio of cm2 per pixel2.
    """
    if reference_pixel_area is not None and reference_pixel_area > 0:
        return PLATE_AREA_CM2 / reference_pixel_area

    total_pixel_area = image_width * image_height
    real_area_cm2 = _FALLBACK_REAL_WIDTH_CM * _FALLBACK_REAL_HEIGHT_CM
    return real_area_cm2 / total_pixel_area
