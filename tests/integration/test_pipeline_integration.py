"""Integration test for the CV pipeline (requires real model weights).

Marked with ``@pytest.mark.integration`` — skipped by CI via
``pytest -m "not integration"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.portion.estimator import PortionEstimator


@pytest.mark.integration
def test_full_cv_pipeline_on_sample_image() -> None:
    """Run PortionEstimator on a real image (no mocks).

    If no images exist in data/raw/ the test is skipped.
    """
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw/ directory does not exist")

    # Find any JPEG/PNG image.
    images = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    if not images:
        pytest.skip("No sample images found in data/raw/")

    image_path = images[0]
    estimator = PortionEstimator()
    results = estimator.estimate(str(image_path))

    # YOLO may or may not detect food — both are acceptable.
    assert isinstance(results, list)

    if results:
        for r in results:
            assert r.estimated_grams > 0
            assert r.label
            assert r.detection_confidence > 0
            assert r.uncertainty_grams >= 0
            assert r.bounding_box is not None
