"""PyTorch Dataset for fresh/rotten produce classification.

Maps directory structure with ``fresh*``/``rotten*`` sub-folders to float labels:
- fresh → 1.0
- rotten → 0.0

Splits are deterministic (seed=42, 80/10/10 train/val/test).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

if TYPE_CHECKING:
    import torch

# CLIP ViT-B/32 normalisation constants.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_VALID_SPLITS = ("train", "val", "test")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SEED = 42
_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test


def _clip_transform() -> transforms.Compose:
    """Return the standard CLIP preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ]
    )


def _parse_label_and_category(folder_name: str) -> tuple[float, str]:
    """Extract (label, category) from a folder name like ``freshapples``.

    Returns ``(1.0, "apples")`` for fresh and ``(0.0, "apples")`` for
    rotten. Raises ``ValueError`` if *folder_name* starts with neither.
    """
    lower = folder_name.lower()
    if lower.startswith("fresh"):
        return 1.0, lower.removeprefix("fresh")
    if lower.startswith("rotten"):
        return 0.0, lower.removeprefix("rotten")
    msg = f"Folder '{folder_name}' does not start with 'fresh' or 'rotten'"
    raise ValueError(msg)


class FreshnessDataset(Dataset):  # type: ignore[type-arg]
    """Image dataset for freshness regression.

    Expects *root* to contain ``fresh*``/``rotten*`` sub-folders.
    *split* must be one of ``"train"``, ``"val"``, ``"test"``.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in _VALID_SPLITS:
            msg = f"split must be one of {_VALID_SPLITS}, got '{split}'"
            raise ValueError(msg)

        self.root = Path(root)
        self.split = split
        self.transform = _clip_transform()

        all_samples: list[tuple[Path, float, str]] = []
        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            try:
                label, category = _parse_label_and_category(subdir.name)
            except ValueError:
                continue

            for img_path in sorted(subdir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    all_samples.append((img_path, label, category))

        rng = random.Random(_SEED)
        rng.shuffle(all_samples)

        n = len(all_samples)
        n_train = int(n * _SPLIT_RATIOS[0])
        n_val = int(n * _SPLIT_RATIOS[1])

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train : n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val :]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, str]:
        path, label, category = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, category
