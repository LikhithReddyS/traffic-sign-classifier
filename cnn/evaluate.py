from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.config import Paths
from utils.gtsrb import load_gtsrb_numpy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained CNN on GTSRB test set.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset root (contains train/ and test/)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model.keras")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = Paths(project_root=project_root)

    dataset_root = Path(args.dataset_root) if args.dataset_root else paths.dataset_root
    model_path = Path(args.model_path) if args.model_path else (paths.cnn_dir / "model.keras")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train CNN first.")

    X_test, y_test = load_gtsrb_numpy(dataset_root, split="test")

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision_macro = float(report["macro avg"]["precision"])
    precision_weighted = float(report["weighted avg"]["precision"])

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    paths.cnn_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.cnn_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
