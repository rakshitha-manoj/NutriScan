"""Download the Fruits Fresh-and-Rotten dataset via kagglehub.

Usage::

    uv run python -m data.download

Environment variables:
    DATA_RAW_DIR  — override the default ``data/raw`` directory.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import kagglehub

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATASET_HANDLE = "sriramr/fruits-fresh-and-rotten-for-classification"
_DEFAULT_RAW_DIR = Path("data/raw")

# Minimum expected sub-directories after a successful download.
_EXPECTED_PREFIXES = ("fresh", "rotten")


def _resolve_raw_dir() -> Path:
    """Return the target directory for raw images."""
    return Path(os.environ.get("DATA_RAW_DIR", str(_DEFAULT_RAW_DIR)))


def _validate_structure(root: Path) -> bool:
    """Check that *root* contains at least one fresh* and one rotten* folder."""
    if not root.is_dir():
        return False
    children = [d.name.lower() for d in root.iterdir() if d.is_dir()]
    has_fresh = any(c.startswith("fresh") for c in children)
    has_rotten = any(c.startswith("rotten") for c in children)
    return has_fresh and has_rotten


def _find_all_image_roots(cache_path: Path) -> list[Path]:
    """Find all directories containing ``fresh*``/``rotten*`` sub-folders.

    The Kaggle dataset nests data inside ``train/`` and ``test/`` dirs.
    We collect every directory that directly contains class sub-folders
    so both splits are merged into a single pool for our own splitting.
    """
    roots: list[Path] = []
    if _validate_structure(cache_path):
        roots.append(cache_path)

    for dirpath, dirnames, _ in os.walk(cache_path):
        dp = Path(dirpath)
        if dp in roots:
            continue
        lower_names = [d.lower() for d in dirnames]
        if any(n.startswith("fresh") for n in lower_names) and any(
            n.startswith("rotten") for n in lower_names
        ):
            roots.append(dp)

    return roots


def _copy_class_images(src: Path, dest: Path) -> int:
    """Copy images from *src* into *dest*, merging with existing files."""
    count = 0
    for img in src.iterdir():
        if img.is_file():
            target = dest / img.name
            if not target.exists():
                shutil.copy2(img, target)
                count += 1
    return count


def download() -> None:
    """Download the dataset and copy class folders into *DATA_RAW_DIR*."""
    raw_dir = _resolve_raw_dir()

    # Skip if data already exists.
    if _validate_structure(raw_dir):
        print(f"[OK] Dataset already present at {raw_dir.resolve()}")
        return

    print(f"[..] Downloading {_DATASET_HANDLE} via kagglehub ...")
    cache_path = Path(kagglehub.dataset_download(_DATASET_HANDLE))
    print(f"[OK] kagglehub cache path: {cache_path}")

    image_roots = _find_all_image_roots(cache_path)
    print(f"[OK] Found {len(image_roots)} image root(s): {image_roots}")

    if not image_roots:
        print(
            f"[FAIL] No fresh*/rotten* folders found under {cache_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Merge class folders from all roots into raw_dir.
    raw_dir.mkdir(parents=True, exist_ok=True)
    total_copied = 0
    for root in image_roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            dest = raw_dir / child.name
            dest.mkdir(exist_ok=True)
            n = _copy_class_images(child, dest)
            total_copied += n
            print(f"  -> {root.name}/{child.name}/: {n} images")

    print(f"[OK] Merged {total_copied} images into {raw_dir.resolve()}")

    # Final validation.
    if not _validate_structure(raw_dir):
        print("[FAIL] Post-copy validation failed.", file=sys.stderr)
        sys.exit(1)

    # Summary.
    total_images = sum(1 for _ in raw_dir.rglob("*") if _.is_file())
    print(f"[OK] Dataset ready -- {total_images} images in {raw_dir.resolve()}")


if __name__ == "__main__":
    download()
