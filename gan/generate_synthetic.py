from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.config import Paths
from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic images for a given class using a trained generator.")
    parser.add_argument("--class_id", type=int, required=True)
    parser.add_argument("--n", type=int, default=2000, help="Number of synthetic images to generate")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator_path", type=str, default=None, help="Path to saved generator.keras")
    args = parser.parse_args()

    seed_everything(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    paths = Paths(project_root=project_root)

    generator_path = Path(args.generator_path) if args.generator_path else (paths.gan_dir / f"class_{args.class_id:02d}" / "generator.keras")
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator not found: {generator_path}. Train GAN first.")

    gen = tf.keras.models.load_model(generator_path, compile=False)

    out_dir = paths.synthetic_dir / f"class_{args.class_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    batch = 256
    remaining = args.n
    idx = 0
    while remaining > 0:
        cur = min(batch, remaining)
        z = tf.random.normal((cur, args.latent_dim))
        imgs = gen(z, training=False).numpy()  # [0,1]
        imgs = (imgs * 255.0).clip(0, 255).astype(np.uint8)
        for i in range(cur):
            img_bgr = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"syn_{idx:06d}.png"), img_bgr)
            idx += 1
        remaining -= cur

    print(f"Wrote {args.n} synthetic images to: {out_dir}")


if __name__ == "__main__":
    main()
