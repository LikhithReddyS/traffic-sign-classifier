from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def dataset_root(self) -> Path:
        # Adjusted for user structure: d:/traffic sign classifier/dataset
        return self.project_root / "dataset"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def cnn_dir(self) -> Path:
        return self.artifacts_dir / "cnn"

    @property
    def synthetic_dir(self) -> Path:
        return self.dataset_root / "synthetic"


IMG_SIZE = 32
NUM_CLASSES = 43

# Augmentation constraints:
# - NO rotation
# - NO horizontal/vertical flip
# Allowed: brightness/contrast, noise (handled in utils/augment.py)
