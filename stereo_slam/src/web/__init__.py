"""Web debugging helpers for StereoSLAM."""

from .app import DEFAULT_WEB_MAP_PATH, create_app, run_web_slam
from .localization_app import (
    DEFAULT_LOCALIZATION_MAP_PATH,
    create_localization_app,
    run_localization_web,
)
from .synthetic import create_synthetic_stereo_frame, create_synthetic_stereo_images

__all__ = [
    "DEFAULT_LOCALIZATION_MAP_PATH",
    "DEFAULT_WEB_MAP_PATH",
    "create_app",
    "create_localization_app",
    "run_web_slam",
    "run_localization_web",
    "create_synthetic_stereo_frame",
    "create_synthetic_stereo_images",
]
