"""Freshness regression model — CLIP VLM features → expiry prediction.

Public API::

    from models.freshness import FreshnessInference, FreshnessPrediction

    engine = FreshnessInference("data/processed/freshness_best.pt")
    result = engine.predict("path/to/image.jpg")
"""

from models.freshness.inference import FreshnessInference, FreshnessPrediction

__all__ = ["FreshnessInference", "FreshnessPrediction"]
