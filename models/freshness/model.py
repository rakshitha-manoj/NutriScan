"""Freshness regression model — MLP head on CLIP embeddings.

Architecture::

    Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
    → Linear(256, 64) → ReLU → Dropout(0.2)
    → Linear(64, 1) → Sigmoid

Output is a scalar in [0, 1] representing freshness (1.0 = fresh, 0.0 = rotten).
Supports MC Dropout inference for uncertainty estimation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_INPUT_DIM = 512
_HIDDEN_1 = 256
_HIDDEN_2 = 64
_DROPOUT_1 = 0.3
_DROPOUT_2 = 0.2


class FreshnessRegressor(nn.Module):
    """MLP regression head mapping CLIP embeddings to freshness scores.

    Args:
        input_dim: Dimensionality of input embeddings (default 512).
        device: Device for the model (default ``"cpu"``).
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, _HIDDEN_1),
            nn.BatchNorm1d(_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(_DROPOUT_1),
            nn.Linear(_HIDDEN_1, _HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(_DROPOUT_2),
            nn.Linear(_HIDDEN_2, 1),
            nn.Sigmoid(),
        )
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Freshness scores of shape ``(B, 1)`` in ``[0, 1]``.
        """
        result: Tensor = self.network(x)
        return result

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_passes: int = 20,
    ) -> tuple[Tensor, Tensor]:
        """MC Dropout inference for uncertainty estimation.

        Enables dropout at eval time, runs *n_passes* stochastic forward
        passes, and returns the mean and standard deviation.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.
            n_passes: Number of Monte Carlo forward passes.

        Returns:
            Tuple of ``(mean, std)`` tensors, each of shape ``(B, 1)``.
        """
        # Enable dropout layers while keeping batchnorm in eval mode.
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        predictions: list[Tensor] = []
        with torch.no_grad():
            for _ in range(n_passes):
                predictions.append(self(x))

        stacked = torch.stack(predictions, dim=0)  # (n_passes, B, 1)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)

        # Restore eval mode.
        self.eval()

        return mean, std
