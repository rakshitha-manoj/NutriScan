"""Train the freshness regression model on preprocessed CLIP embeddings.

Usage::

    uv run python -m models.freshness.train

Loads ``.pt`` files from ``data/processed/``, trains
:class:`~models.freshness.model.FreshnessRegressor`, and saves the best
checkpoint to ``data/processed/freshness_best.pt``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from models.freshness.model import FreshnessRegressor

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)  # noqa: NPY002
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
_LR = 1e-3
_WEIGHT_DECAY = 1e-4
_EPOCHS = 50
_BATCH_SIZE = 64
_PATIENCE = 10

_DEFAULT_PROCESSED_DIR = Path("data/processed")


def _resolve_dir() -> Path:
    return Path(os.environ.get("DATA_PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))


def _load_split(processed_dir: Path, split: str) -> TensorDataset:
    emb = torch.load(processed_dir / f"embeddings_{split}.pt", weights_only=True)
    lab = torch.load(processed_dir / f"labels_{split}.pt", weights_only=True)
    return TensorDataset(emb, lab.unsqueeze(1))


def _train_one_epoch(
    model: FreshnessRegressor,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(x)
        loss: Tensor = criterion(pred, y)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def _evaluate(
    model: FreshnessRegressor,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss: Tensor = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def train() -> None:
    """Run the full training loop."""
    processed_dir = _resolve_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = _load_split(processed_dir, "train")
    val_ds = _load_split(processed_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=_BATCH_SIZE)

    input_dim = train_ds.tensors[0].shape[1]
    model = FreshnessRegressor(input_dim=input_dim, device=str(device))
    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=_EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = processed_dir / "freshness_best.pt"

    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("-" * 30)

    for epoch in range(1, _EPOCHS + 1):
        t_loss = _train_one_epoch(model, train_loader, criterion, optimiser, device)
        v_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>5}  {t_loss:>10.6f}  {v_loss:>10.6f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": {"input_dim": input_dim},
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= _PATIENCE:
                print(
                    f"\n[STOP] Early stopping at epoch {epoch} "
                    f"(no improvement for {_PATIENCE} epochs)"
                )
                break

    print(f"\n[OK] Best val loss: {best_val_loss:.6f}")
    print(f"[OK] Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
