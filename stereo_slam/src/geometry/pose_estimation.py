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
        """
        使用 PnP 算法估计相机位姿
        
        Args:
            object_points: 3D 点坐标 (N, 3)
            image_points: 2D 图像点坐标 (N, 2)
            
        Returns:
            camera_pose: 相机位姿 (4x4 矩阵)
            num_inliers: 内点数量
        """
        if len(object_points) < 10 or len(image_points) < 10:
            return self.camera_pose, 0
        
        # PnP 求解相机位姿
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.K,
            self.distortion_coeffs,
            iterationsCount=100,
            reprojectionError=self.ransac_reproj_threshold,
            confidence=0.99
        )
        
        if not success:
            return self.camera_pose, 0
        
        num_inliers = len(inliers) if inliers is not None else 0
        inliers_ratio = num_inliers / len(object_points)
        
        # 转换位姿
        R, _ = cv2.Rodrigues(rvec)
        
        # 构建相机位姿
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = tvec.flatten()
        
        # 更新相机位姿（累积）
        self.camera_pose = self.camera_pose @ new_pose
        
        return self.camera_pose, num_inliers
    
    def get_pose(self) -> np.ndarray:
        """获取当前相机位姿"""
        return self.camera_pose.copy()
    
    def reset(self):
        """重置位姿估计器"""
        self.camera_pose = np.eye(4)