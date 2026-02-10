from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image_rgb(path: str | Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_to_32(img_rgb: np.ndarray, size: int = 32) -> np.ndarray:
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)


def normalize_01(img_rgb: np.ndarray) -> np.ndarray:
    return (img_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def preprocess_image(img_rgb: np.ndarray, size: int = 32) -> np.ndarray:
    img_rgb = resize_to_32(img_rgb, size=size)
    img_rgb = normalize_01(img_rgb)
    return img_rgb


def preprocess_frame_for_model(frame_bgr: np.ndarray, size: int = 32) -> np.ndarray:
    # frame comes from OpenCV camera in BGR
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = preprocess_image(frame_rgb, size=size)
    return frame_rgb
