from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow not installed yet or optional in some contexts.
        pass
