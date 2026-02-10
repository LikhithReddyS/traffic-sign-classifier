from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.config import Paths
from utils.gtsrb import class_distribution, load_gtsrb_numpy, minority_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GTSRB class distribution and minority classes.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset/GTSRB")
    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.25,
        help="Classes at or below this quantile of class counts are flagged as minority.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = Paths(project_root=project_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root else paths.dataset_root

    print(f"Loading dataset from: {dataset_root}")
    try:
        X_train, y_train = load_gtsrb_numpy(dataset_root, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dist = class_distribution(y_train)

    counts = list(dist.values())
    if not counts:
        print("No classes found.")
        return

    print(f"Total training images: {len(X_train)}")
    print(f"Num classes present: {len(dist)}")
    print(f"Min count: {min(counts)} | Max count: {max(counts)}")

    minorities = minority_classes(y_train, threshold_quantile=args.threshold_quantile)
    print(f"Minority classes (quantile={args.threshold_quantile}): {minorities}")

    # Print top/bottom counts for quick review
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1])
    print("\nBottom 10 classes by count:")
    for cls, c in sorted_items[:10]:
        print(f"  class {cls:02d}: {c}")

    print("\nTop 10 classes by count:")
    for cls, c in sorted_items[-10:][::-1]:
        print(f"  class {cls:02d}: {c}")


if __name__ == "__main__":
    main()
