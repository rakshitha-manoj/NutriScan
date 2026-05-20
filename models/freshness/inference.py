"""Inference module for the freshness regression model.

Usage::

    from models.freshness import FreshnessInference, FreshnessPrediction

    engine = FreshnessInference("data/processed/freshness_best.pt")
    result = engine.predict("path/to/image.jpg")
    print(result.freshness_score, result.label, result.uncertainty)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.freshness.extractor import CLIPExtractor
from models.freshness.model import FreshnessRegressor

# CLIP ViT-B/32 normalisation constants.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class FreshnessPrediction:
    """Result of a single freshness prediction.

    ``freshness_score`` ranges from 0.0 (rotten) to 1.0 (fresh).
    ``uncertainty`` is the standard deviation from MC Dropout.
    ``label`` is ``"fresh"`` if score >= 0.5, else ``"rotten"``.
    ``confidence`` is ``1.0 - uncertainty``, clipped to ``[0, 1]``.
    """

    freshness_score: float
    uncertainty: float
    label: str
    confidence: float


def _build_transform() -> transforms.Compose:
    """Preprocessing pipeline matching CLIP input expectations."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ]
    )


class FreshnessInference:
    """End-to-end freshness inference: image -> CLIP -> MLP -> prediction.

    Loads a trained checkpoint from *checkpoint_path*, runs on *device*
    (auto-detected by default), and uses *n_mc_passes* MC Dropout
    forward passes for uncertainty estimation. Raises ``ValueError``
    if the checkpoint file does not exist.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        n_mc_passes: int = 20,
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            msg = f"Checkpoint not found: {ckpt_path}"
            raise ValueError(msg)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.n_mc_passes = n_mc_passes
        self.transform = _build_transform()

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        config: dict[str, int] = checkpoint["config"]
        self.model = FreshnessRegressor(
            input_dim=config["input_dim"],
            device=device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.extractor = CLIPExtractor(device=device)

    def predict(self, image_path: str | Path) -> FreshnessPrediction:
        """Run inference on a single image.

        Accepts a path to a JPEG/PNG file and returns a
        :class:`FreshnessPrediction` with score, uncertainty, label,
        and confidence.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        embedding = self.extractor.extract(tensor)  # (1, 512)
        embedding = embedding.to(self.device)

        mean, std = self.model.predict_with_uncertainty(embedding, n_passes=self.n_mc_passes)

        score = float(mean.item())
        unc = float(std.item())

        return FreshnessPrediction(
            freshness_score=score,
            uncertainty=unc,
            label="fresh" if score >= 0.5 else "rotten",
            confidence=max(0.0, min(1.0, 1.0 - unc)),
        )
