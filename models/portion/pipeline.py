"""Combined freshness + portion estimation pipeline.

Runs YOLOv8 detection, geometric portion estimation, and CLIP-based
freshness inference on each detected item from a single fridge image.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from models.freshness.inference import FreshnessInference
from models.portion.estimator import PortionEstimator

if TYPE_CHECKING:
    from db.models import BoundingBox


@dataclass(frozen=True)
class ProduceItem:
    """Merged result for a single detected food item.

    Combines portion data (label, bounding box, grams, uncertainty)
    with freshness data (score, uncertainty, label) from both the
    YOLO and CLIP pipelines.
    """

    label: str
    bounding_box: BoundingBox
    detection_confidence: float
    estimated_grams: float
    uncertainty_grams: float
    freshness_score: float
    freshness_uncertainty: float
    freshness_label: str


class PortionPipeline:
    """End-to-end pipeline: image -> detection + grams + freshness.

    Requires a trained freshness model checkpoint at
    *freshness_checkpoint*.
    """

    def __init__(self, freshness_checkpoint: str | Path) -> None:
        self._estimator = PortionEstimator()
        self._freshness = FreshnessInference(freshness_checkpoint)

    def run(self, image_path: str | Path) -> list[ProduceItem]:
        """Run the full pipeline on a single image.

        For each detected food item, crops the bounding box region,
        runs freshness inference, and returns a list of
        :class:`ProduceItem` with portion and freshness data merged.
        """
        image_path = Path(image_path)
        portions = self._estimator.estimate(image_path)

        if not portions:
            return []

        img = Image.open(image_path).convert("RGB")
        items: list[ProduceItem] = []

        for portion in portions:
            bb = portion.bounding_box
            f_score = 0.5
            f_unc = 0.5
            f_label = "unknown"

            try:
                crop = img.crop(
                    (
                        int(bb.x_min),
                        int(bb.y_min),
                        int(bb.x_max),
                        int(bb.y_max),
                    )
                )

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    crop.save(tmp, format="JPEG")
                    tmp_path = Path(tmp.name)

                pred = self._freshness.predict(tmp_path)
                f_score = pred.freshness_score
                f_unc = pred.uncertainty
                f_label = pred.label

                tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass  # fallback to defaults

            items.append(
                ProduceItem(
                    label=portion.label,
                    bounding_box=portion.bounding_box,
                    detection_confidence=portion.detection_confidence,
                    estimated_grams=portion.estimated_grams,
                    uncertainty_grams=portion.uncertainty_grams,
                    freshness_score=f_score,
                    freshness_uncertainty=f_unc,
                    freshness_label=f_label,
                )
            )

        return items
