from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cnn.model import build_cnn
from utils.augment import augment_allowed
from utils.config import IMG_SIZE, NUM_CLASSES, Paths
from utils.gtsrb import class_distribution, load_gtsrb_numpy
from utils.image_ops import preprocess_image, read_image_rgb
from utils.seed import seed_everything


def _load_synthetic(paths: Paths) -> Tuple[np.ndarray, np.ndarray]:
    syn_root = paths.synthetic_dir
    if not syn_root.exists():
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    Xs = []
    ys = []

    for class_dir in sorted([p for p in syn_root.iterdir() if p.is_dir()]):
        # class folder name: class_XX
        try:
            class_id = int(class_dir.name.split("_")[1])
        except Exception:
            continue

        for img_path in class_dir.glob("*.png"):
            img = read_image_rgb(img_path)
            img = preprocess_image(img, size=IMG_SIZE)
            Xs.append(img)
            ys.append(class_id)

    if not Xs:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)


def _make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 20_000), reshuffle_each_iteration=True)

        def _aug(x, label):
            x_np = x.numpy()
            x_np = augment_allowed(x_np)
            return x_np.astype(np.float32), label

        ds = ds.map(
            lambda x, label: tf.py_function(_aug, inp=[x, label], Tout=[tf.float32, tf.int64]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(lambda x, label: (tf.ensure_shape(x, (IMG_SIZE, IMG_SIZE, 3)), tf.ensure_shape(label, ())),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on real + GAN-generated synthetic images.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset root (contains train/ and test/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    paths = Paths(project_root=project_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root else paths.dataset_root

    X_train, y_train = load_gtsrb_numpy(dataset_root, split="train")
    X_test, y_test = load_gtsrb_numpy(dataset_root, split="test")

    # Train/val split from the official training set (keeps official test split untouched).
    X_tr_real, X_val_real, y_tr_real, y_val_real = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=args.seed,
        stratify=y_train,
    )

    dist = class_distribution(y_tr_real)
    min_count = min(dist.values())
    max_count = max(dist.values())
    print(f"Train split class counts: min={min_count}, max={max_count}")

    X_syn, y_syn = _load_synthetic(paths)

    if len(X_syn) > 0:
        X_tr = np.concatenate([X_tr_real, X_syn], axis=0)
        y_tr = np.concatenate([y_tr_real, y_syn], axis=0)
        print(f"Loaded synthetic: {len(X_syn)} images")
    else:
        X_tr, y_tr = X_tr_real, y_tr_real
        print("No synthetic images found; training on real only.")

    model = build_cnn(num_classes=NUM_CLASSES, img_size=IMG_SIZE)

    train_ds = _make_dataset(X_tr, y_tr, batch_size=args.batch_size, training=True)
    val_ds = _make_dataset(X_val_real, y_val_real, batch_size=args.batch_size, training=False)

    paths.cnn_dir.mkdir(parents=True, exist_ok=True)
    model_path = paths.cnn_dir / "model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    model.save(model_path)

    with open(paths.cnn_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    # Save split info (useful for academic review)
    split_info = {
        "train_real": int(len(X_tr_real)),
        "val_real": int(len(X_val_real)),
        "test_real": int(len(X_test)),
        "synthetic": int(len(X_syn)),
        "seed": int(args.seed),
        "val_fraction": 0.15,
        "note": "Validation split is drawn from official training set. Official test split is reserved for final evaluation.",
    }
    with open(paths.cnn_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved CNN model to: {model_path}")


if __name__ == "__main__":
    main()
