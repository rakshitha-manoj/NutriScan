"""Unit tests for portion estimation pipeline.

All tests use mocked YOLO output -- no real images or model downloads
needed in CI.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from db.models import BoundingBox
from models.portion.categories import FOOD_CATEGORIES
from models.portion.estimator import PortionEstimate, PortionEstimator
from models.portion.pipeline import ProduceItem
from models.portion.reference import detect_reference_object, estimate_pixel_to_cm2_ratio

# ---------------------------------------------------------------------------
# Helpers -- build mock YOLO results
# ---------------------------------------------------------------------------


def _make_mock_box(
    label: str,
    conf: float,
    xyxy: tuple[float, float, float, float],
) -> MagicMock:
    """Create a mock ultralytics box object."""
    box = MagicMock()
    box.conf = [conf]
    box.cls = [0]  # overridden via names dict
    xyxy_tensor = MagicMock()
    xyxy_tensor.tolist.return_value = list(xyxy)
    box.xyxy = [xyxy_tensor]
    box._label = label
    return box


def _make_mock_result(
    boxes: list[MagicMock],
    names: dict[int, str] | None = None,
) -> MagicMock:
    """Create a mock ultralytics result object."""
    result = MagicMock()
    if names is None:
        names = {}
        for i, b in enumerate(boxes):
            b.cls = [i]
            names[i] = b._label
    result.names = names
    result.boxes = boxes
    return result


def _mock_yolo_call(
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
) -> MagicMock:
    """Build a mock YOLO callable returning the given detections."""
    boxes = [_make_mock_box(lb, c, xy) for lb, c, xy in detections]
    result = _make_mock_result(boxes)
    yolo = MagicMock()
    yolo.return_value = [result]
    return yolo


# ---------------------------------------------------------------------------
# PortionEstimator tests
# ---------------------------------------------------------------------------


class TestPortionEstimator:
    """Tests for PortionEstimator.estimate()."""

    @patch("PIL.Image.open")
    def test_estimate_returns_portion_estimates(self, mock_open: MagicMock) -> None:
        """estimate() returns a list of PortionEstimate objects."""
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_open.return_value = mock_img

        est = PortionEstimator()
        est._yolo = _mock_yolo_call(
            [
                ("apple", 0.9, (100, 100, 200, 200)),
            ]
        )

        results = est.estimate("fake.jpg")
        assert len(results) == 1
        assert isinstance(results[0], PortionEstimate)

    @patch("PIL.Image.open")
    def test_estimated_grams_positive(self, mock_open: MagicMock) -> None:
        """Estimated grams should be positive for known categories."""
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_open.return_value = mock_img

        est = PortionEstimator()
        est._yolo = _mock_yolo_call(
            [
                ("apple", 0.9, (100, 100, 200, 200)),
            ]
        )

        results = est.estimate("fake.jpg")
        assert results[0].estimated_grams > 0

    @patch("PIL.Image.open")
    def test_estimated_grams_within_bounds(self, mock_open: MagicMock) -> None:
        """Grams should be clamped to [1, grams_per_unit * 1.5]."""
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_open.return_value = mock_img

        est = PortionEstimator()
        est._yolo = _mock_yolo_call(
            [
                ("apple", 0.9, (10, 10, 600, 600)),
            ]
        )

        results = est.estimate("fake.jpg")
        meta = FOOD_CATEGORIES["apple"]
        assert results[0].estimated_grams >= 1.0
        assert results[0].estimated_grams <= meta.grams_per_unit * 1.5

    @patch("PIL.Image.open")
    def test_uncertainty_minimum_floor(self, mock_open: MagicMock) -> None:
        """Uncertainty should be at least 5g."""
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_open.return_value = mock_img

        est = PortionEstimator()
        est._yolo = _mock_yolo_call(
            [
                ("apple", 0.9, (100, 100, 200, 200)),
            ]
        )

        results = est.estimate("fake.jpg")
        assert results[0].uncertainty_grams >= 5.0

    @patch("PIL.Image.open")
    def test_unknown_labels_filtered(self, mock_open: MagicMock) -> None:
        """YOLO labels not in FOOD_CATEGORIES are excluded."""
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_open.return_value = mock_img

        est = PortionEstimator()
        est._yolo = _mock_yolo_call(
            [
                ("person", 0.9, (100, 100, 200, 200)),
                ("car", 0.8, (300, 300, 400, 400)),
            ]
        )

        results = est.estimate("fake.jpg")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Reference object tests
# ---------------------------------------------------------------------------


class TestReferenceObject:
    """Tests for reference object detection and ratio estimation."""

    def test_no_reference_returns_none(self) -> None:
        """Returns None when no reference object is in the boxes."""
        boxes = [
            BoundingBox(x_min=10, y_min=10, x_max=60, y_max=60),
        ]
        labels = ["apple"]
        assert detect_reference_object(boxes, labels) is None

    def test_bowl_reference_returns_float(self) -> None:
        """Returns pixel area when a bowl is detected."""
        boxes = [
            BoundingBox(x_min=10, y_min=10, x_max=210, y_max=210),
        ]
        labels = ["bowl"]
        result = detect_reference_object(boxes, labels)
        assert result is not None
        assert result == pytest.approx(200 * 200)

    def test_ratio_fallback_when_no_reference(self) -> None:
        """Falls back to heuristic when reference is None."""
        ratio = estimate_pixel_to_cm2_ratio(None, 640, 480)
        expected = (60.0 * 45.0) / (640 * 480)
        assert ratio == pytest.approx(expected)

    def test_ratio_uses_plate_when_reference_present(self) -> None:
        """Uses plate area when a reference object is detected."""
        plate_area = math.pi * (26.0 / 2) ** 2
        ref_pixels = 40000.0
        ratio = estimate_pixel_to_cm2_ratio(ref_pixels, 640, 480)
        assert ratio == pytest.approx(plate_area / ref_pixels)


# ---------------------------------------------------------------------------
# PortionPipeline tests
# ---------------------------------------------------------------------------


class TestPortionPipeline:
    """Tests for the combined pipeline."""

    @patch("models.portion.pipeline.FreshnessInference")
    @patch("models.portion.pipeline.PortionEstimator")
    def test_run_returns_produce_items(
        self,
        mock_estimator_cls: MagicMock,
        mock_freshness_cls: MagicMock,
    ) -> None:
        """run() returns ProduceItem with portion + freshness."""
        # Mock PortionEstimator.
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = [
            PortionEstimate(
                label="apple",
                bounding_box=BoundingBox(
                    x_min=100,
                    y_min=100,
                    x_max=200,
                    y_max=200,
                ),
                detection_confidence=0.9,
                estimated_grams=150.0,
                uncertainty_grams=10.0,
                pixel_to_cm2_ratio=0.01,
            ),
        ]
        mock_estimator_cls.return_value = mock_estimator

        # Mock FreshnessInference.
        mock_freshness = MagicMock()
        mock_pred = MagicMock()
        mock_pred.freshness_score = 0.85
        mock_pred.uncertainty = 0.03
        mock_pred.label = "fresh"
        mock_freshness.predict.return_value = mock_pred
        mock_freshness_cls.return_value = mock_freshness

        from models.portion.pipeline import PortionPipeline

        pipeline = PortionPipeline("fake_checkpoint.pt")
        with patch("PIL.Image.open") as mock_img_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_img.crop.return_value = mock_img
            mock_img_open.return_value = mock_img

            results = pipeline.run("fake_image.jpg")

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, ProduceItem)
        assert item.label == "apple"
        assert item.estimated_grams == 150.0
        assert item.freshness_score == 0.85
        assert item.freshness_label == "fresh"
