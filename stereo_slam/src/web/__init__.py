"""Web debugging helpers for StereoSLAM."""

from .app import DEFAULT_WEB_MAP_PATH, create_app, run_web_slam
from .synthetic import create_synthetic_stereo_frame, create_synthetic_stereo_images

__all__ = [
    "DEFAULT_WEB_MAP_PATH",
    "create_app",
    "run_web_slam",
    "create_synthetic_stereo_frame",
    "create_synthetic_stereo_images",
]
