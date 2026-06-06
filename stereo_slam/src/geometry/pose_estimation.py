"""
位姿估计模块
使用 PnP 算法估计相机位姿
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class PoseEstimator:
    """
    位姿估计器
    使用 PnP 算法估计相机位姿
    """
    
    def __init__(
        self,
        K: np.ndarray,
        distortion_coeffs: np.ndarray = None,
        min_inliers_ratio: float = 0.15,
        ransac_reproj_threshold: float = 3.0
    ):
        """
        初始化位姿估计器
        
        Args:
            K: 相机内参矩阵
            distortion_coeffs: 畸变系数，默认为零
            min_inliers_ratio: 最小内点比例
            ransac_reproj_threshold: RANSAC 重投影阈值
        """
        self.K = K.copy()
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros((4, 1))
        self.min_inliers_ratio = min_inliers_ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold
        
        # 当前相机位姿
        self.camera_pose = np.eye(4)
        
    def estimate_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        
        # PnP usually needs at least 4-6 points for RANSAC to be effective
        if len(object_points) < 6:
            return self.camera_pose, 0
        
        # 1. PnP Solving
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            self.K,
            self.distortion_coeffs,
            iterationsCount=100,
            reprojectionError=self.ransac_reproj_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE # Stable for most cases
        )
        
        if not success or inliers is None:
            return self.camera_pose, 0

        # 2. Quality Check
        num_inliers = len(inliers)
        if num_inliers / len(object_points) < self.min_inliers_ratio:
            return self.camera_pose, 0
        
        # 3. Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # 4. Construct the World-to-Camera Matrix (T_cw)
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = tvec.flatten()
        
        # 5. Calculate Camera-to-World (T_wc) 
        # This represents the camera's actual position in your map
        T_wc = np.linalg.inv(T_cw)
        
        self.camera_pose = T_wc
        
        return self.camera_pose, num_inliers
    
    def get_pose(self) -> np.ndarray:
        """获取当前相机位姿"""
        return self.camera_pose.copy()
    
    def reset(self):
        """重置位姿估计器"""
        self.camera_pose = np.eye(4)