"""
三角测量模块
用于通过立体匹配计算 3D 点位置
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class StereoTriangulator:
    """立体三角测量器"""
    
    def __init__(
        self,
        baseline: float = 0.065,  # 基线距离，单位：米
        focal_length: float = 1000.0,  # 焦距
        principal_point: Tuple[float, float] = (1280.0, 360.0)  # 主点
    ):
        """
        初始化立体三角测量器
        
        Args:
            baseline: 左右相机基线距离（米）
            focal_length: 焦距（像素）
            principal_point: 主点坐标 (cx, cy)
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.cx, self.cy = principal_point
        
    def triangulate_point(
        self,
        left_point: Tuple[float, float],
        right_point: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """
        通过立体匹配计算 3D 点位置
        
        Args:
            left_point: 左图坐标 (u_left, v_left)
            right_point: 右图坐标 (u_right, v_right)
            
        Returns:
            3D 位置 [x, y, z] 或 None（如果无法计算）
        """
        u_l, v_l = left_point
        u_r, v_r = right_point
        
        # 计算视差
        disparity = u_l - u_r
        
        # 避免除以零或负视差
        if disparity <= 0:
            return None
            
        # 使用简单的三角测量公式
        # Z = f * B / d
        z = self.focal_length * self.baseline / disparity
        
        # X = (u - cx) * Z / f
        x = (u_l - self.cx) * z / self.focal_length
        
        # Y = (v - cy) * Z / f
        y = (v_l - self.cy) * z / self.focal_length
        
        return np.array([x, y, z])
    
    def triangulate_matches(
        self,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        三角测量所有匹配点
        
        Args:
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            matches: 匹配对列表
            
        Returns:
            (feature_id, 3D_position) 列表
        """
        results = []
        
        for match in matches:
            left_pt = left_keypoints[match.queryIdx].pt
            right_pt = right_keypoints[match.trainIdx].pt
            
            position = self.triangulate_point(left_pt, right_pt)
            
            if position is not None:
                # 使用左图特征点 ID
                feature_id = int(match.queryIdx)
                results.append((feature_id, position))
                
        return results