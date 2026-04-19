"""
地图管理模块
管理 3D 地图和关键帧
改进版：支持观测驱动的点管理和点过滤
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from .keyframe import KeyFrame
from .point import Point3D


class Map:
    """3D 地图类"""
    
    def __init__(self, device_id: int = 0, min_observations: int = 2):
        """
        初始化地图
        
        Args:
            device_id: 设备 ID
            min_observations: 最小观测次数（低于此值的点会被过滤）
        """
        self.device_id = device_id
        self.min_observations = min_observations
        
        # 3D 点：{id: Point3D}
        self.points: Dict[int, Point3D] = {}
        
        # 关键帧
        self.keyframes: Dict[int, KeyFrame] = {}
        self.frame_counter = 0
        self.created_at = datetime.now().isoformat()
        
    def add_3d_point(
        self,
        position: np.ndarray,
        color: Optional[np.ndarray] = None,
        observation_ids: Optional[List[int]] = None
    ) -> int:
        """
        添加 3D 点到地图
        
        Args:
            position: 3D 位置 [x, y, z]
            color: 颜色 [b, g, r]
            observation_ids: 观测该点的帧 ID 列表
            
        Returns:
            点的唯一 ID
        """
        point_id = len(self.points)
        
        point = Point3D(
            position=position.copy(),
            color=color.copy() if color is not None else None,
            observation_count=len(observation_ids) if observation_ids else 1,
            last_seen_frame=observation_ids[-1] if observation_ids else 0,
            observation_ids=observation_ids or []
        )
        
        self.points[point_id] = point
        return point_id
    
    def update_3d_point(
        self,
        point_id: int,
        position: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        add_observation: Optional[int] = None,
        use_weighted_average: bool = True,
        update_weight: float = 0.3
    ):
        """
        更新 3D 点信息（使用加权平均）
        
        Args:
            point_id: 点 ID
            position: 新的位置
            color: 新的颜色
            add_observation: 添加观测帧 ID
            use_weighted_average: 是否使用加权平均
            update_weight: 更新权重
        """
        if point_id not in self.points:
            return
            
        point = self.points[point_id]
        
        if position is not None:
            # 使用加权平均更新位置
            point.add_observation(
                frame_id=add_observation or point.last_seen_frame,
                position=position,
                weight=update_weight,
                use_weighted_average=use_weighted_average
            )
            
        if color is not None:
            point.color = color.copy()
    
    def remove_3d_point(self, point_id: int):
        """移除 3D 点"""
        if point_id in self.points:
            del self.points[point_id]
    
    def cull_insecure_points(self) -> int:
        """
        删除不安全的点（观测次数不足）
        
        Returns:
            删除的点数
        """
        points_to_remove = []
        
        for point_id, point in self.points.items():
            if point.should_cull(self.min_observations):
                points_to_remove.append(point_id)
        
        for point_id in points_to_remove:
            del self.points[point_id]
        
        return len(points_to_remove)
    
    def filter_points_by_depth(self, min_depth: float = 0.5, 
                                max_depth: float = 20.0) -> int:
        """
        过滤深度范围外的点
        
        Args:
            min_depth: 最小深度
            max_depth: 最大深度
            
        Returns:
            删除的点数
        """
        points_to_remove = []
        
        for point_id, point in self.points.items():
            depth = np.linalg.norm(point.position)
            if depth < min_depth or depth > max_depth:
                points_to_remove.append(point_id)
        
        for point_id in points_to_remove:
            del self.points[point_id]
        
        return len(points_to_remove)
    
    def add_keyframe(
        self,
        frame_id: int,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        left_descriptors: np.ndarray,
        right_descriptors: np.ndarray,
        camera_pose: Optional[np.ndarray] = None
    ) -> KeyFrame:
        """
        添加关键帧
        
        Args:
            frame_id: 帧 ID
            left_image: 左图像
            right_image: 右图像
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            left_descriptors: 左图描述子
            right_descriptors: 右图描述子
            camera_pose: 相机位姿
            
        Returns:
            创建的关键帧对象
        """
        keyframe = KeyFrame(
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
        
        self.keyframes[frame_id] = keyframe
        self.frame_counter = max(self.frame_counter, frame_id + 1)
        
        return keyframe
    
    def get_keyframe(self, frame_id: int) -> Optional[KeyFrame]:
        """获取指定帧的关键帧"""
        return self.keyframes.get(frame_id)
    
    def get_all_keyframes(self) -> List[KeyFrame]:
        """获取所有关键帧"""
        return list(self.keyframes.values())
    
    def get_3d_points_array(self) -> np.ndarray:
        """获取所有 3D 点位置数组"""
        if not self.points:
            return np.array([])
        return np.array([p.position for p in self.points.values()])
    
    def get_3d_points_colors(self) -> Optional[np.ndarray]:
        """获取所有 3D 点颜色数组"""
        colors = [
            p.color for p in self.points.values() 
            if p.color is not None
        ]
        if not colors:
            return None
        return np.array(colors, dtype=np.uint8)
    
    def get_statistics(self) -> Dict:
        """获取地图统计信息"""
        positions = self.get_3d_points_array()
        
        stats = {
            "device_id": self.device_id,
            "num_points": len(self.points),
            "num_keyframes": len(self.keyframes),
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat()
        }
        
        if len(positions) > 0:
            stats["bounding_box"] = {
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist()
            }
            stats["center"] = positions.mean(axis=0).tolist()
            
        return stats
    
    def save_to_file(self, filepath: str):
        """保存地图到文件"""
        stats = self.get_statistics()
        
        # 转换点数据为可序列化格式
        points_data = {}
        for point_id, point in self.points.items():
            points_data[str(point_id)] = {
                "position": point.position.tolist(),
                "color": point.color.tolist() if point.color is not None else None,
                "observation_count": point.observation_count,
                "observation_ids": point.observation_ids,
                "last_seen_frame": point.last_seen_frame
            }
        
        stats["points"] = points_data
        stats["keyframe_ids"] = list(self.keyframes.keys())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Map':
        """从文件加载地图"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        map_obj = cls(
            device_id=data.get("device_id", 0),
            min_observations=data.get("min_observations", 2)
        )
        
        # 加载点
        for point_id_str, point_data in data.get("points", {}).items():
            point_id = int(point_id_str)
            position = np.array(point_data["position"])
            color = np.array(point_data["color"]) if point_data.get("color") else None
            
            point = Point3D(
                position=position,
                color=color,
                observation_count=point_data.get("observation_count", 1),
                last_seen_frame=point_data.get("last_seen_frame", 0),
                observation_ids=point_data.get("observation_ids", [])
            )
            
            map_obj.points[point_id] = point
        
        map_obj.frame_counter = data.get("num_keyframes", 0)
        
        return map_obj