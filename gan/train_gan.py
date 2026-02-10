from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from gan.dcgan import DCGAN, DCGANConfig, build_discriminator, build_generator
from utils.config import IMG_SIZE, Paths
from utils.gtsrb import load_gtsrb_numpy
from utils.seed import seed_everything


def _make_dataset(X: np.ndarray, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.shuffle(min(len(X), 10_000), reshuffle_each_iteration=True)
    # If the class has fewer samples than batch_size, drop_remainder=True would create an empty dataset.
    drop_remainder = len(X) >= batch_size
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _save_samples(generator: tf.keras.Model, out_dir: Path, latent_dim: int, n: int = 64) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    z = tf.random.normal((n, latent_dim))
    imgs = generator(z, training=False).numpy()  # [0,1]

    import cv2

    for i in range(n):
        img = (imgs[i] * 255.0).clip(0, 255).astype(np.uint8)
        # RGB -> BGR for cv2.imwrite
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"sample_{i:03d}.png"), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DCGAN on a specific (minority) GTSRB class.")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset root (contains train/ and test/)")
    parser.add_argument("--class_id", type=int, required=True, help="GTSRB class id (0-42)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    paths = Paths(project_root=project_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root else paths.dataset_root

    X_train, y_train = load_gtsrb_numpy(dataset_root, split="train")
    Xc = X_train[y_train == args.class_id]
    if len(Xc) == 0:
        raise ValueError(
            f"No samples found for class_id={args.class_id}. "
            "Check that the dataset is present and class_id is within 0-42."
        )
    if len(Xc) < 200:
        print(f"Warning: class {args.class_id} has only {len(Xc)} images; GAN may underfit.")

    effective_batch = min(int(args.batch_size), int(len(Xc)))
    ds = _make_dataset(Xc, batch_size=effective_batch)

    cfg = DCGANConfig(img_size=IMG_SIZE, channels=3, latent_dim=args.latent_dim)
    gen = build_generator(cfg)
    disc = build_discriminator(cfg)
    gan = DCGAN(gen, disc, cfg)
    gan.compile()

    out_dir = paths.gan_dir / f"class_{args.class_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "generator.keras"

    class SampleCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                _save_samples(gen, out_dir / "samples" / f"epoch_{epoch+1:04d}", cfg.latent_dim, n=32)
                gen.save(ckpt_path)

    gan.fit(ds, epochs=args.epochs, callbacks=[SampleCallback()])
    gen.save(ckpt_path)

    print(f"Saved generator to: {ckpt_path}")


if __name__ == "__main__":
    main()
