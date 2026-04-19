"""
关键帧数据结构
"""

import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class KeyFrame:
    """关键帧数据结构"""
    frame_id: int
    timestamp: float
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    left_keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    right_keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    left_descriptors: Optional[np.ndarray] = None
    right_descriptors: Optional[np.ndarray] = None
    camera_pose: Optional[np.ndarray] = None  # 4x4 变换矩阵
    matched_3d_points: List[int] = field(default_factory=list)  # 关联的 3D 点 ID
    
    @classmethod
    def create(
        cls,
        frame_id: int,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        left_descriptors: np.ndarray,
        right_descriptors: np.ndarray,
        camera_pose: Optional[np.ndarray] = None
    ) -> 'KeyFrame':
        """创建关键帧的工厂方法"""
        return cls(
            frame_id=frame_id,
            timestamp=datetime.now().timestamp(),
            left_image=left_image,
            right_image=right_image,
            left_keypoints=left_keypoints,
            right_keypoints=right_keypoints,
            left_descriptors=left_descriptors,
            right_descriptors=right_descriptors,
            camera_pose=camera_pose
        )