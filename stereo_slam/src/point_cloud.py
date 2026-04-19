"""
3D 点云模块
用于计算 3D 点坐标和管理点云数据
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class Point3D:
    """3D 点数据结构"""
    position: np.ndarray  # [x, y, z]
    color: Optional[np.ndarray] = None  # [b, g, r]
    observation_count: int = 0  # 观测次数
    last_seen_frame: int = 0  # 最后看到的帧号


class PointCloud:
    """3D 点云管理器"""
    
    def __init__(self):
        """初始化点云"""
        self.points: Dict[int, Point3D] = {}  # feature_id -> Point3D
        self.point_id_counter = 0
        
    def add_point(self, position: np.ndarray, color: Optional[np.ndarray] = None, 
                  feature_id: Optional[int] = None) -> int:
        """
        添加 3D 点到点云
        
        Args:
            position: 3D 位置 [x, y, z]
            color: 颜色 [b, g, r]
            feature_id: 特征 ID，如果为 None 则自动生成
            
        Returns:
            点的 ID
        """
        if feature_id is None:
            feature_id = self.point_id_counter
            self.point_id_counter += 1
            
        point = Point3D(
            position=position.copy(),
            color=color.copy() if color is not None else None,
            observation_count=1,
            last_seen_frame=0
        )
        
        self.points[feature_id] = point
        return feature_id
    
    def update_point(self, feature_id: int, position: np.ndarray, 
                     color: Optional[np.ndarray] = None):
        """
        更新已存在的 3D 点
        
        Args:
            feature_id: 特征 ID
            position: 新的 3D 位置
            color: 新的颜色
        """
        if feature_id in self.points:
            point = self.points[feature_id]
            point.position = position.copy()
            if color is not None:
                point.color = color.copy()
            point.observation_count += 1
            
    def remove_point(self, feature_id: int):
        """移除 3D 点"""
        if feature_id in self.points:
            del self.points[feature_id]
    
    def get_all_points(self) -> List[Point3D]:
        """获取所有点"""
        return list(self.points.values())
    
    def get_point_positions(self) -> np.ndarray:
        """获取所有点的位置数组"""
        if not self.points:
            return np.array([])
        return np.array([p.position for p in self.points.values()])
    
    def get_point_colors(self) -> Optional[np.ndarray]:
        """获取所有点的颜色数组"""
        colors = [p.color for p in self.points.values() if p.color is not None]
        if not colors:
            return None
        return np.array(colors)
    
    def get_statistics(self) -> Dict:
        """获取点云统计信息"""
        if not self.points:
            return {
                "num_points": 0,
                "bounding_box": None,
                "center": None
            }
            
        positions = self.get_point_positions()
        return {
            "num_points": len(self.points),
            "bounding_box": {
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist()
            },
            "center": positions.mean(axis=0).tolist()
        }


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