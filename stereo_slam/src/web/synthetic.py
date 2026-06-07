"""Synthetic stereo frames for tests and fallback debugging."""

import cv2
import numpy as np


def create_synthetic_stereo_images(width: int = 1280, height: int = 720):
    """Create a simple synthetic rectified stereo pair."""
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    np.random.seed(42)

    for _ in range(500):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        radius = np.random.randint(3, 10)
        color = (
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255),
        )
        cv2.circle(left_img, (x, y), radius, color, -1)

    right_img = left_img.copy()
    right_img[:, :-30] = left_img[:, 30:]
    right_img[:, -30:] = 0
    return left_img, right_img


def create_synthetic_stereo_frame(
    width: int = 2560,
    height: int = 720,
    eye_order: str = "left-right",
):
    """Create one side-by-side synthetic stereo frame."""
    eye_width = width // 2
    left_img, right_img = create_synthetic_stereo_images(eye_width, height)
    if eye_order == "right-left":
        return np.hstack((right_img, left_img))
    return np.hstack((left_img, right_img))

