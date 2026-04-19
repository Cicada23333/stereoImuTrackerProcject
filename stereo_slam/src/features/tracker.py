"""
特征跟踪模块
使用光流法跟踪特征点
"""

import cv2
import numpy as np
from typing import Tuple, List


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