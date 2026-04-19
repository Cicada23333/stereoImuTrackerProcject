"""
地图更新器
处理新特征与已有地图点的关联
"""

import numpy as np
from typing import Tuple


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