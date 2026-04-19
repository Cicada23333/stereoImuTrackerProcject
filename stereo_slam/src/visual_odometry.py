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


class FeatureTracker:
    """
    特征跟踪器
    使用光流法跟踪特征点
    """
    
    def __init__(
        self,
        max_features: int = 1000,
        quality_level: float = 0.01,
        min_distance: int = 10
    ):
        """
        初始化特征跟踪器
        
        Args:
            max_features: 最大特征点数量
            quality_level: 质量等级
            min_distance: 最小距离
        """
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # 光流参数
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.prev_keypoints = []
        
    def detect_features(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        检测特征点
        
        Args:
            gray: 灰度图像
            
        Returns:
            keypoints: 检测到的特征点
            mask: 特征点掩码
        """
        # 使用 goodFeaturesToTrack 检测角点
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        
        if corners is None:
            return [], np.zeros(gray.shape, dtype=np.uint8)
        
        corners = corners.reshape(-1, 2)
        keypoints = [cv2.KeyPoint(p[0], p[1], 5) for p in corners]
        
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for kp in keypoints:
            cv2.circle(mask, (int(kp.pt[0]), int(kp.pt[1])), 3, 255, -1)
        
        return keypoints, mask
    
    def track_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_keypoints: List[cv2.KeyPoint]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        跟踪特征点
        
        Args:
            prev_gray: 前一帧灰度图
            curr_gray: 当前帧灰度图
            prev_keypoints: 前一帧特征点
            
        Returns:
            prev_pts: 前一帧特征点坐标
            curr_pts: 当前帧特征点坐标
            status: 跟踪状态
        """
        if len(prev_keypoints) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 使用 Lucas-Kanade 光流
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2),
            None,
            **self.lk_params
        )
        
        status = status.flatten()
        
        # 过滤有效点
        valid_indices = np.where(status == 1)[0]
        prev_pts = np.array([prev_keypoints[i].pt for i in valid_indices], dtype=np.float32)
        curr_pts = np.array([curr_pts[i][0] for i in valid_indices], dtype=np.float32)
        
        return prev_pts, curr_pts, status
    
    def update(self, gray: np.ndarray):
        """更新跟踪器"""
        self.prev_gray = gray


class MapUpdater:
    """
    地图更新器
    处理新特征与已有地图点的关联
    """
    
    def __init__(self, distance_threshold: float = 0.1):
        """
        初始化地图更新器
        
        Args:
            distance_threshold: 距离阈值，用于判断是否匹配已有地图点
        """
        self.distance_threshold = distance_threshold
        
    def find_matching_points(
        self,
        new_points: np.ndarray,
        existing_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        查找新点与已有点的匹配
        
        Args:
            new_points: 新 3D 点坐标 (N, 3)
            existing_points: 已存在 3D 点坐标 (M, 3)
            
        Returns:
            new_indices: 新点索引
            existing_indices: 已有点索引
        """
        if len(existing_points) == 0:
            return np.array([]), np.array([])
        
        # 计算距离矩阵
        distances = np.linalg.norm(
            new_points[:, np.newaxis, :] - existing_points[np.newaxis, :, :],
            axis=2
        )
        
        # 找到最近的匹配
        new_indices = []
        existing_indices = []
        
        for i, dist_row in enumerate(distances):
            min_dist = np.min(dist_row)
            if min_dist < self.distance_threshold:
                closest_idx = np.argmin(dist_row)
                new_indices.append(i)
                existing_indices.append(closest_idx)
        
        return np.array(new_indices), np.array(existing_indices)
    
    def update_map_points(
        self,
        existing_points: np.ndarray,
        new_observations: np.ndarray,
        match_indices: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        更新地图点位置（使用平均位置）
        
        Args:
            existing_points: 已存在 3D 点坐标
            new_observations: 新观测的 3D 点坐标
            match_indices: 匹配索引 (new_indices, existing_indices)
            
        Returns:
            updated_points: 更新后的 3D 点坐标
        """
        new_indices, existing_indices = match_indices
        
        if len(new_indices) == 0:
            return existing_points
        
        updated_points = existing_points.copy()
        
        # 对每个匹配的点对，计算平均位置
        for new_idx, existing_idx in zip(new_indices, existing_indices):
            new_pos = new_observations[new_idx]
            existing_pos = existing_points[existing_idx]
            
            # 简单的平均
            updated_points[existing_idx] = (new_pos + existing_pos) / 2
        
        return updated_points