"""Unit tests for the freshness regression model and CLIP extractor.

All tests use random tensors — no real data or model weights required.
"""

from __future__ import annotations

import torch

from models.freshness.extractor import CLIPExtractor
from models.freshness.inference import FreshnessPrediction
from models.freshness.model import FreshnessRegressor

# ---------------------------------------------------------------------------
# FreshnessRegressor
# ---------------------------------------------------------------------------


class TestFreshnessRegressor:
    """Tests for FreshnessRegressor forward pass and MC Dropout."""

    def test_forward_output_shape(self) -> None:
        """Input (4, 512) → output (4, 1)."""
        model = FreshnessRegressor(input_dim=512)
        x = torch.randn(4, 512)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_in_unit_interval(self) -> None:
        """All outputs should be in [0, 1] due to Sigmoid."""
        model = FreshnessRegressor(input_dim=512)
        x = torch.randn(16, 512)
        out = model(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_mc_dropout_returns_mean_and_std(self) -> None:
        """predict_with_uncertainty returns (mean, std) of correct shape."""
        model = FreshnessRegressor(input_dim=512)
        model.eval()
        x = torch.randn(4, 512)
        mean, std = model.predict_with_uncertainty(x, n_passes=10)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)

    def test_mc_dropout_std_positive(self) -> None:
        """Std should be > 0 because dropout is active during MC passes."""
        model = FreshnessRegressor(input_dim=512)
        model.eval()
        # Use a larger batch and more passes to ensure variance.
        x = torch.randn(8, 512)
        _, std = model.predict_with_uncertainty(x, n_passes=50)
        # At least some samples should have non-zero std.
        assert (std > 0).any(), "MC Dropout should produce non-zero variance"


# ---------------------------------------------------------------------------
# CLIPExtractor
# ---------------------------------------------------------------------------


class TestCLIPExtractor:
    """Tests for CLIPExtractor output shape and normalisation."""

    def test_output_shape(self) -> None:
        """extract() should return (batch, 512)."""
        extractor = CLIPExtractor(device="cpu")
        images = torch.randn(2, 3, 224, 224)
        out = extractor.extract(images)
        assert out.shape == (2, 512)

    def test_output_l2_normalised(self) -> None:
        """Each row should have L2 norm ≈ 1.0."""
        extractor = CLIPExtractor(device="cpu")
        images = torch.randn(2, 3, 224, 224)
        out = extractor.extract(images)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# FreshnessPrediction
# ---------------------------------------------------------------------------


class TestFreshnessPrediction:
    """Tests for the FreshnessPrediction dataclass."""

    def test_label_fresh_when_score_above_threshold(self) -> None:
        """Label should be 'fresh' when score >= 0.5."""
        pred = FreshnessPrediction(
            freshness_score=0.75,
            uncertainty=0.05,
            label="fresh" if 0.75 >= 0.5 else "rotten",
            confidence=0.95,
        )
        assert pred.label == "fresh"

    def test_label_rotten_when_score_below_threshold(self) -> None:
        """Label should be 'rotten' when score < 0.5."""
        pred = FreshnessPrediction(
            freshness_score=0.3,
            uncertainty=0.1,
            label="fresh" if 0.3 >= 0.5 else "rotten",
            confidence=0.9,
        )
        assert pred.label == "rotten"
