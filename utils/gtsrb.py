from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.image_ops import preprocess_image, read_image_rgb
from utils.config import IMG_SIZE


@dataclass(frozen=True)
class GTSRBPaths:
    dataset_root: Path

    @property
    def train_images_root(self) -> Path:
        return self.dataset_root / "train" / "Final_Training" / "Images"

    @property
    def test_images_root(self) -> Path:
        return self.dataset_root / "test" / "Final_Test" / "Images"

    @property
    def test_csv(self) -> Path:
        # In official GTSRB, this file is usually: Final_Test/Images/GT-final_test.csv
        # In this user's structure, it's at dataset root. We check both.
        p = self.dataset_root / "GT-final_test.csv"
        if p.exists():
            return p
        return self.test_images_root / "GT-final_test.csv"


def _load_train_split(paths: GTSRBPaths) -> Tuple[List[Path], List[int]]:
    img_paths: List[Path] = []
    labels: List[int] = []

    if not paths.train_images_root.exists():
        raise FileNotFoundError(
            f"Training folder not found: {paths.train_images_root}. "
            "Download GTSRB and place it under dataset/train/Final_Training/Images/."
        )

    # Each class is a folder 00000..00042 with GT-xxxxx.csv
    for class_dir in sorted([p for p in paths.train_images_root.iterdir() if p.is_dir()]):
        csv_candidates = list(class_dir.glob("GT-*.csv"))
        if not csv_candidates:
            continue
        gt_csv = csv_candidates[0]
        df = pd.read_csv(gt_csv, sep=";")
        # Columns typically include Filename and ClassId
        for _, row in df.iterrows():
            rel = str(row["Filename"])
            # Assuming row["Filename"] ends with .ppm, swap it to .jpg
            if rel.lower().endswith('.ppm'):
                rel = rel[:-4] + '.jpg'
            img_path = class_dir / rel
            img_paths.append(img_path)
            labels.append(int(row["ClassId"]))

    if not img_paths:
        raise RuntimeError("No training images found. Check your GTSRB directory layout.")

    return img_paths, labels


def _load_test_split(paths: GTSRBPaths) -> Tuple[List[Path], List[int]]:
    if not paths.test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {paths.test_csv}. "
            "Ensure GTSRB Final_Test/Images contains GT-final_test.csv."
        )

    df = pd.read_csv(paths.test_csv, sep=";")
    img_paths: List[Path] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        rel = str(row["Filename"])
        if rel.lower().endswith('.ppm'):
            rel = rel[:-4] + '.jpg'
        img_path = paths.test_images_root / rel
        img_paths.append(img_path)
        labels.append(int(row["ClassId"]))

    return img_paths, labels


def load_gtsrb_numpy(dataset_root: str | Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Loads GTSRB to numpy arrays.

    Returns:
        X: float32 (N, 32, 32, 3) in [0,1]
        y: int64 (N,)

    Note: No rotation/flips are applied here.
    """
    dataset_root = Path(dataset_root)
    paths = GTSRBPaths(dataset_root=dataset_root)

    if split == "train":
        img_paths, labels = _load_train_split(paths)
    elif split == "test":
        img_paths, labels = _load_test_split(paths)
    else:
        raise ValueError("split must be 'train' or 'test'")

    X = np.zeros((len(img_paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    for i, p in enumerate(img_paths):
        img = read_image_rgb(p)
        img = preprocess_image(img, size=IMG_SIZE)
        X[i] = img

    return X, y


def class_distribution(y: np.ndarray) -> Dict[int, int]:
    uniq, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def minority_classes(y: np.ndarray, threshold_quantile: float = 0.25) -> List[int]:
    dist = class_distribution(y)
    counts = np.array(list(dist.values()), dtype=np.int64)
    cutoff = int(np.quantile(counts, threshold_quantile))
    return sorted([cls for cls, c in dist.items() if c <= cutoff])
