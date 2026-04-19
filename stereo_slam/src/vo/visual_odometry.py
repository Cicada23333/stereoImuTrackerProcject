"""
视觉里程计模块
用于估计相机位姿和跟踪特征点
支持增量式地图更新
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class VisualOdometry:
    """
    视觉里程计类
    使用 PnP 算法估计相机位姿，支持增量式地图更新
    """
    
    def __init__(
        self,
        K: np.ndarray,
        distortion_coeffs: np.ndarray = None,
        min_inliers_ratio: float = 0.15,
        ransac_reproj_threshold: float = 3.0
    ):
        """
        初始化视觉里程计
        
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
        
        # 前一帧的信息
        self.prev_keypoints: List[cv2.KeyPoint] = []
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_3d_points: Optional[np.ndarray] = None
        
        # 当前相机位姿
        self.camera_pose = np.eye(4)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def initialize(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        three_d_points: np.ndarray
    ):
        """
        初始化视觉里程计
        
        Args:
            keypoints: 当前帧特征点
            descriptors: 描述子
            three_d_points: 对应的 3D 点
        """
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_3d_points = three_d_points.copy()
        self.camera_pose = np.eye(4)
        self.logger.info("VisualOdometry initialized")
        
    def update(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        """
        更新相机位姿
        
        Args:
            keypoints: 当前帧特征点
            descriptors: 描述子
            
        Returns:
            camera_pose: 更新的相机位姿 (4x4 矩阵)
            num_matches: 匹配数量
            num_inliers: 内点数量
        """
        if len(self.prev_keypoints) < 10 or self.prev_descriptors is None:
            self.logger.warning("Not enough previous keypoints for VO update")
            return self.camera_pose, 0, 0
            
        if len(keypoints) < 10 or descriptors is None:
            self.logger.warning("Not enough current keypoints for VO update")
            return self.camera_pose, 0, 0
        
        # 1. 特征匹配
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            self.logger.warning(f"Not enough good matches: {len(good_matches)}")
            return self.camera_pose, len(good_matches), 0
        
        # 2. 提取匹配点坐标和对应的 3D 点
        prev_points = []
        curr_points = []
        matched_3d_points = []
        
        for match in good_matches:
            prev_pt = self.prev_keypoints[match.queryIdx].pt
            curr_pt = keypoints[match.trainIdx].pt
            
            # 检查对应的 3D 点是否存在
            if match.queryIdx < len(self.prev_3d_points):
                prev_points.append(prev_pt)
                curr_points.append(curr_pt)
                matched_3d_points.append(self.prev_3d_points[match.queryIdx])
        
        if len(prev_points) < 10:
            self.logger.warning(f"Not enough matched points with 3D: {len(prev_points)}")
            return self.camera_pose, len(good_matches), 0
        
        prev_points = np.array(prev_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)
        matched_3d_points = np.array(matched_3d_points, dtype=np.float32)
        
        # 3. PnP 求解相机位姿
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_3d_points,
            curr_points,
            self.K,
            self.distortion_coeffs,
            iterationsCount=100,
            reprojectionError=self.ransac_reproj_threshold,
            confidence=0.99
        )
        
        if not success:
            self.logger.warning("PnP failed")
            return self.camera_pose, len(good_matches), 0
        
        num_inliers = len(inliers) if inliers is not None else 0
        inliers_ratio = num_inliers / len(prev_points)
        
        self.logger.debug(f"PnP: {num_inliers}/{len(prev_points)} inliers ({inliers_ratio:.1%})")
        
        # 检查内点比例
        if inliers_ratio < self.min_inliers_ratio:
            self.logger.warning(f"Low inlier ratio: {inliers_ratio:.1%}")
        
        # 4. 转换位姿
        R, _ = cv2.Rodrigues(rvec)
        
        # 构建新的相机位姿
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = tvec.flatten()
        
        # 更新相机位姿（累积）
        self.camera_pose = self.camera_pose @ new_pose
        
        # 更新历史数据
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.camera_pose, len(good_matches), num_inliers
    
    def get_pose(self) -> np.ndarray:
        """获取当前相机位姿"""
        return self.camera_pose.copy()
    
    def reset(self):
        """重置视觉里程计"""
        self.prev_keypoints = []
        self.prev_descriptors = None
        self.prev_3d_points = None
        self.camera_pose = np.eye(4)