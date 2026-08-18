from __future__ import annotations

import numpy as np


import cv2
import random

def random_affine(img: np.ndarray,
                  translate_range: float = 0.1,
                  zoom_range: float = 0.1) -> np.ndarray:
    """Applies random affine transformation (shift, zoom only -- NO rotation).

    Rotation is intentionally excluded: it is not semantics-preserving for
    directional traffic signs (e.g. turn-left/turn-right, keep-left/keep-right).
    """
    h, w = img.shape[:2]

    # Translation
    tx = np.random.uniform(-translate_range, translate_range) * w
    ty = np.random.uniform(-translate_range, translate_range) * h

    # Zoom
    zoom = np.random.uniform(1.0 - zoom_range, 1.0 + zoom_range)

    # Construct affine matrix (scale only, no rotation)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0.0, zoom)

    # Add translation to the matrix
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


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
    """Allowed augmentation (geometric + photometric)."""
    x = img
    
    # Geometric first (wrap/reflect handling works best on clean image)
    if np.random.rand() < p:
        x = random_affine(x)
        
    # Photometric
    if np.random.rand() < p:
        x = random_brightness_contrast(x)
        
    if np.random.rand() < p:
        x = add_gaussian_noise(x)
        
    return x
