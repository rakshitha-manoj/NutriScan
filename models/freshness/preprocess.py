"""Batch-extract CLIP embeddings and save as .pt files.

Usage::

    uv run python -m models.freshness.preprocess

Reads images via :class:`FreshnessDataset`, encodes them with
:class:`CLIPExtractor`, and writes embeddings / labels / category
metadata to ``data/processed/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.freshness.dataset import FreshnessDataset
from models.freshness.extractor import CLIPExtractor

_DEFAULT_RAW_DIR = Path("data/raw")
_DEFAULT_PROCESSED_DIR = Path("data/processed")
_BATCH_SIZE = 32
_SPLITS = ("train", "val", "test")


def _resolve_dirs() -> tuple[Path, Path]:
    raw = Path(os.environ.get("DATA_RAW_DIR", str(_DEFAULT_RAW_DIR)))
    processed = Path(os.environ.get("DATA_PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))
    return raw, processed


def _output_exists(processed_dir: Path) -> bool:
    """Check whether all output files already exist."""
    for split in _SPLITS:
        if not (processed_dir / f"embeddings_{split}.pt").exists():
            return False
        if not (processed_dir / f"labels_{split}.pt").exists():
            return False
    return (processed_dir / "category_names.json").exists()


def _collate(
    batch: list[tuple[torch.Tensor, float, str]],
) -> tuple[torch.Tensor, list[float], list[str]]:
    """Custom collate: stack images, keep labels/categories as lists."""
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    categories = [b[2] for b in batch]
    return images, labels, categories


def preprocess() -> None:
    """Extract CLIP embeddings for all splits and save to disk."""
    raw_dir, processed_dir = _resolve_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)

    if _output_exists(processed_dir):
        print("[OK] Preprocessed files already exist -- skipping extraction.")
        return

    extractor = CLIPExtractor()
    all_categories: set[str] = set()

    for split in _SPLITS:
        print(f"\n[..] Processing {split} split ...")
        dataset = FreshnessDataset(root=raw_dir, split=split)
        loader = DataLoader(
            dataset,
            batch_size=_BATCH_SIZE,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
        )

        embeddings_list: list[torch.Tensor] = []
        labels_list: list[float] = []

        for images, labels, categories in loader:
            emb = extractor.extract(images)
            embeddings_list.append(emb)
            labels_list.extend(labels)
            all_categories.update(categories)

        embeddings = torch.cat(embeddings_list, dim=0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

        torch.save(embeddings, processed_dir / f"embeddings_{split}.pt")
        torch.save(labels_tensor, processed_dir / f"labels_{split}.pt")

        print(
            f"  [OK] {split}: {embeddings.shape[0]} samples, "
            f"embedding shape {tuple(embeddings.shape)}"
        )

    sorted_cats = sorted(all_categories)
    cat_map = {i: name for i, name in enumerate(sorted_cats)}
    cat_path = processed_dir / "category_names.json"
    cat_path.write_text(json.dumps(cat_map, indent=2))
    print(f"\n[OK] Saved category mapping ({len(cat_map)} categories) to {cat_path}")
    print("[OK] Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
