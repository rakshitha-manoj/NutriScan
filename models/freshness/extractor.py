"""CLIP ViT-B/32 feature extractor using open-clip-torch.

Produces L2-normalised 512-dim embeddings from preprocessed image batches.
Model is frozen (no grad) at all times.
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"
_EMBED_DIM = 512


class CLIPExtractor:
    """Extract frozen CLIP visual embeddings.

    Args:
        device: Device to run on (``"cpu"`` or ``"cuda"``).
            Defaults to CUDA if available, else CPU.
    """

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model, _, _ = open_clip.create_model_and_transforms(_MODEL_NAME, pretrained=_PRETRAINED)
        self.model: nn.Module = model.visual
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters.
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, images: Tensor) -> Tensor:
        """Return L2-normalised 512-dim embeddings for a batch of images.

        Args:
            images: Preprocessed image batch of shape ``(B, 3, 224, 224)``.

        Returns:
            Float32 tensor of shape ``(B, 512)``, L2-normalised per row.
        """
        images = images.to(self.device)
        features: Tensor = self.model(images)
        features = features.float()
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu()

    @property
    def embed_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return _EMBED_DIM
