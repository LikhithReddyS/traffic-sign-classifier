"""GTSRB training-set reader (Python 3).

This file replaces the legacy Python-2 / matplotlib sample.

It is NOT required by the main project pipeline (which uses utils/gtsrb.py),
but is kept as a small, runnable utility for academic review/debugging.

Expected rootpath (recommended): dataset/train/Final_Training/Images
which contains class folders 00000..00042 and GT-xxxxx.csv files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project root to sys.path so utils.* imports work when running directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.config import IMG_SIZE
from utils.image_ops import preprocess_image, read_image_rgb


def readTrafficSigns(rootpath: str | Path, as_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray] | Tuple[List[np.ndarray], List[int]]:
    """Reads GTSRB *training* images and labels.

    Args:
        rootpath: Path to `train/Final_Training/Images` (contains 00000..00042)
        as_numpy: If True returns (X, y) arrays; else returns python lists.

    Returns:
        If as_numpy:
            X: float32 (N, 32, 32, 3) in [0,1]
            y: int64 (N,)
        Else:
            images: list of float32 (32,32,3) arrays
            labels: list of int
    """

    root = Path(rootpath)
    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_id in range(43):
        class_dir = root / f"{class_id:05d}"
        gt_csv = class_dir / f"GT-{class_id:05d}.csv"
        if not gt_csv.exists():
            # If a class folder is missing, skip rather than crash.
            continue

        with gt_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)  # skip header
            for row in reader:
                filename = row[0]
                label = int(row[7])

                img = read_image_rgb(class_dir / filename)
                img = preprocess_image(img, size=IMG_SIZE)

                images.append(img)
                labels.append(label)

    if as_numpy:
        if not images:
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
        X = np.stack(images).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y

    return images, labels


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read GTSRB training images/labels (Python 3 utility).")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent / "train" / "Final_Training" / "Images"),
        help="Path to dataset/train/Final_Training/Images",
    )
    args = parser.parse_args()

    X, y = readTrafficSigns(args.root, as_numpy=True)
    print(f"Loaded: X={X.shape}, y={y.shape}")
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Classes present: {len(unique)}")
        print(f"Min class count: {int(counts.min())}, Max class count: {int(counts.max())}")


if __name__ == "__main__":
    _main()
