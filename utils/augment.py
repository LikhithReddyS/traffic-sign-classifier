from __future__ import annotations

import numpy as np


def random_brightness_contrast(img: np.ndarray, brightness_delta: float = 0.15, contrast_range=(0.85, 1.15)) -> np.ndarray:
    """Applies brightness/contrast without any geometric transforms.

    img: float32 RGB in [0, 1]
    """
    x = img
    # brightness
    b = np.random.uniform(-brightness_delta, brightness_delta)
    x = x + b
    # contrast
    c = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean
    return np.clip(x, 0.0, 1.0)


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Adds Gaussian noise; no rotation/flips."""
    noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def augment_allowed(img: np.ndarray, p: float = 0.8) -> np.ndarray:
    """Allowed augmentation only (brightness/contrast + noise)."""
    x = img
    if np.random.rand() < p:
        x = random_brightness_contrast(x)
    if np.random.rand() < p:
        x = add_gaussian_noise(x)
    return x
