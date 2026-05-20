"""Portion estimator using YOLOv8 bounding boxes and geometric depth cues.

Converts bounding-box pixel area to real-world gram estimates using a
reference-object calibration or fallback heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from db.models import BoundingBox
from models.portion.categories import FOOD_CATEGORIES, YOLO_LABEL_TO_CATEGORY
from models.portion.reference import (
    detect_reference_object,
    estimate_pixel_to_cm2_ratio,
)

_MIN_UNCERTAINTY_GRAMS: float = 5.0
_ASPECT_UNCERTAINTY_SCALE: float = 0.3


@dataclass(frozen=True)
class PortionEstimate:
    """Result for a single detected food item.

    Includes the food label, bounding box, YOLO confidence, estimated
    weight in grams, 1-sigma uncertainty, and the pixel-to-cm² ratio
    used for the estimate.
    """

    label: str
    bounding_box: BoundingBox
    detection_confidence: float
    estimated_grams: float
    uncertainty_grams: float
    pixel_to_cm2_ratio: float


def _bbox_width(bb: BoundingBox) -> float:
    return bb.x_max - bb.x_min


def _bbox_height(bb: BoundingBox) -> float:
    return bb.y_max - bb.y_min


class PortionEstimator:
    """Estimate food portions from a fridge image using YOLOv8n.

    YOLO is initialised lazily on first call to :meth:`estimate` so that
    tests can mock it without triggering a model download.
    """

    def __init__(self) -> None:
        self._yolo: Any | None = None

    def _get_yolo(self) -> Any:
        """Lazily load YOLOv8n model."""
        if self._yolo is None:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._yolo = YOLO("yolov8n.pt")
        return self._yolo

    def estimate(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.25,
    ) -> list[PortionEstimate]:
        """Run detection and portion estimation on a single image.

        Accepts a path to a JPEG/PNG *image_path* and filters detections
        below *confidence_threshold*. Returns a list of
        :class:`PortionEstimate` for each recognised food item.
        """
        image_path = Path(image_path)
        img = Image.open(image_path)
        img_w, img_h = img.size

        yolo = self._get_yolo()
        results = yolo(str(image_path), verbose=False)

        all_labels: list[str] = []
        all_boxes: list[BoundingBox] = []
        all_confs: list[float] = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                all_labels.append(label)
                all_boxes.append(bbox)
                all_confs.append(conf)

        ref_area = detect_reference_object(all_boxes, all_labels)
        ratio = estimate_pixel_to_cm2_ratio(ref_area, img_w, img_h)

        estimates: list[PortionEstimate] = []
        for bbox, conf, label in zip(all_boxes, all_confs, all_labels, strict=True):
            category_key = YOLO_LABEL_TO_CATEGORY.get(label.lower())
            if category_key is None:
                continue

            meta = FOOD_CATEGORIES[category_key]
            w = _bbox_width(bbox)
            h = _bbox_height(bbox)

            # Geometric estimation.
            pixel_area = w * h
            area_cm2 = pixel_area * ratio
            volume_cm3 = area_cm2 * meta.typical_height_cm
            grams = volume_cm3 * meta.density_g_per_cm3

            max_grams = meta.grams_per_unit * 1.5
            grams = max(1.0, min(grams, max_grams))

            # Uncertainty from aspect ratio deviation.
            aspect_ratio = w / max(h, 1e-6)
            aspect_dev = abs(aspect_ratio - meta.expected_aspect_ratio) / max(
                meta.expected_aspect_ratio, 1e-6
            )
            uncertainty = max(
                _MIN_UNCERTAINTY_GRAMS,
                grams * aspect_dev * _ASPECT_UNCERTAINTY_SCALE,
            )

            estimates.append(
                PortionEstimate(
                    label=category_key,
                    bounding_box=bbox,
                    detection_confidence=conf,
                    estimated_grams=round(grams, 1),
                    uncertainty_grams=round(uncertainty, 1),
                    pixel_to_cm2_ratio=ratio,
                )
            )

        return estimates
