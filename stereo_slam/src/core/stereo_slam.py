"""Stereo SLAM system assembly."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ..features import FeatureExtractor, StereoMatcher
from ..geometry import StereoTriangulator
from ..map import Map
from ..vo import VisualOdometry
from .config import CameraConfig, SLAMConfig
from .frame_processing import FrameProcessingMixin
from .image_input import StereoImageInputMixin
from .map_association import MapAssociationMixin
from .map_io import MapIOMixin
from .observations import ObservationMixin
from .pose import PoseUtilsMixin


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class StereoSLAM(
    StereoImageInputMixin,
    FrameProcessingMixin,
    ObservationMixin,
    MapAssociationMixin,
    MapIOMixin,
    PoseUtilsMixin,
):
    """OpenCV ORB based stereo SLAM prototype."""

    def __init__(
        self,
        device_id: int = 0,
        baseline: float = 0.065,
        focal_length: Optional[float] = None,
        image_width: int = 1280,
        image_height: int = 720,
        stereo_width: Optional[int] = None,
        fov_horizontal: float = 100.0,
        eye_order: str = "left-right",
        auto_save_path: Optional[Union[str, Path]] = None,
        debug_mode: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.device_id = device_id
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height
        self.stereo_width = stereo_width or image_width * 2
        if eye_order not in ("left-right", "right-left"):
            raise ValueError('eye_order must be "left-right" or "right-left"')
        self.eye_order = eye_order
        self.auto_save_path = Path(auto_save_path) if auto_save_path else None
        self.debug_mode = debug_mode

        focal_from_fov = image_width / (2 * np.tan(np.deg2rad(fov_horizontal / 2)))
        self.focal_length = float(focal_length) if focal_length is not None else float(focal_from_fov)
        self.principal_point = (image_width / 2, image_height / 2)

        self.config = SLAMConfig(
            device_id=device_id,
            debug_mode=debug_mode,
            camera=CameraConfig(
                image_width=image_width,
                image_height=image_height,
                fov_horizontal=fov_horizontal,
                baseline=baseline,
                focal_length=self.focal_length,
            ),
        )

        self.logger = logger or logging.getLogger(f"stereo_slam_{device_id}")
        self.feature_extractor = FeatureExtractor(n_features=self.config.feature.n_features)
        self.stereo_matcher = StereoMatcher(
            ratio_threshold=self.config.matching.ratio_threshold,
            cross_check=self.config.matching.cross_check,
        )
        self.triangulator = StereoTriangulator(
            baseline=baseline,
            focal_length=self.focal_length,
            principal_point=self.principal_point,
        )
        self.map = Map(
            device_id=device_id,
            min_observations=self.config.map.min_observations,
        )

        self.K = np.array(
            [
                [self.focal_length, 0, self.principal_point[0]],
                [0, self.focal_length, self.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.camera_pose = np.eye(4)
        self.visual_odometry = VisualOdometry(self.K)

        self.current_3d_points: Optional[np.ndarray] = None
        self.current_left_keypoints: List[cv2.KeyPoint] = []
        self.current_left_descriptors: Optional[np.ndarray] = None
        self.last_observations: List[Dict] = []

        self.logger.info(f"StereoSLAM initialized for device {device_id}")
        self.logger.info(f"  Baseline: {baseline}m")
        self.logger.info(f"  Focal length: {self.focal_length:.2f} pixels")
        self.logger.info(f"  FOV: {fov_horizontal} deg")
        self.logger.info(f"  Eye image size: {image_width}x{image_height}")
        self.logger.info(f"  Side-by-side image size: {self.stereo_width}x{image_height}")
        self.logger.info(f"  Eye order: {self.eye_order}")
