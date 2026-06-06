"""
几何工具函数
提供常用的几何计算功能
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class GeometryUtils:
    """几何工具类"""
    
    @staticmethod
    def project_3d_to_2d_with_depth(
        points_3d: np.ndarray,
        camera_pose: np.ndarray,
        K: np.ndarray,
        image_shape: Tuple[int, int],
        return_indices: bool = False
    ):
        """
        将世界坐标 3D 点投影到图像平面，并保留相机坐标深度。
        
        Args:
            points_3d: 3D 点坐标 (N, 3)
            camera_pose: 相机位姿矩阵 (4x4)，从世界坐标系到相机坐标系的变换
            K: 相机内参矩阵
            image_shape: 图像形状
            
        Returns:
            投影后的 (x, y, z_cam)。return_indices=True 时同时返回原始点索引。
        """
        if len(points_3d) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            if return_indices:
                return empty, np.empty((0,), dtype=np.int64)
            return empty
        
        # camera_pose 是从世界到相机的变换
        # X_cam = R * X_world + t
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        
        # 将 3D 点从世界坐标系转换到相机坐标系
        points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
        
        # 只保留在相机前方的点
        valid_mask = points_cam[:, 2] > 0.1
        if not np.any(valid_mask):
            empty = np.empty((0, 3), dtype=np.float64)
            if return_indices:
                return empty, np.empty((0,), dtype=np.int64)
            return empty
        
        valid_indices = np.flatnonzero(valid_mask)
        points_cam_valid = points_cam[valid_mask]
        
        # 投影到图像平面
        points_2d = (K @ points_cam_valid.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        h, w = image_shape[:2]
        in_image = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        )

        projected = np.column_stack((points_2d[in_image], points_cam_valid[in_image, 2]))
        if return_indices:
            return projected, valid_indices[in_image]
        
        return projected

    @staticmethod
    def project_3d_to_2d(
        points_3d: np.ndarray,
        camera_pose: np.ndarray,
        K: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """将世界坐标 3D 点投影到 2D 图像平面。"""
        projected = GeometryUtils.project_3d_to_2d_with_depth(
            points_3d, camera_pose, K, image_shape
        )
        return projected[:, :2]
    
    @staticmethod
    def find_nearby_point(
        position: np.ndarray,
        existing_points: dict,
        threshold: float = 0.05
    ) -> Optional[int]:
        """
        查找附近已存在的 3D 点
        
        Args:
            position: 新的 3D 位置
            existing_points: 已存在的 3D 点字典 {id: Point3D} 或 {id: {position, ...}}
            threshold: 距离阈值
            
        Returns:
            最近的点 ID，如果没有则返回 None
        """
        if not existing_points:
            return None
        
        closest_id = None
        closest_distance = float('inf')
        
        for point_id, point_data in existing_points.items():
            # 支持 Point3D 对象和字典两种格式
            if hasattr(point_data, 'position'):
                # Point3D 对象
                existing_pos = np.array(point_data.position)
            else:
                # 字典格式
                existing_pos = np.array(point_data["position"])
            
            distance = np.linalg.norm(position - existing_pos)
            
            # 只返回非常接近的点
            if distance < threshold and distance < closest_distance:
                closest_id = point_id
                closest_distance = distance
        
        # 只有当距离非常近时才认为存在附近点
        if closest_distance < threshold:
            return closest_id
        
        return None
    
    @staticmethod
    def create_pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        从旋转矩阵和平移向量创建 4x4 位姿矩阵
        
        Args:
            R: 3x3 旋转矩阵
            t: 3x1 平移向量
            
        Returns:
            4x4 位姿矩阵
        """
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        return pose
    
    @staticmethod
    def pose_to_se3(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 4x4 位姿矩阵分解为旋转矩阵和平移向量
        
        Args:
            pose: 4x4 位姿矩阵
            
        Returns:
            R: 3x3 旋转矩阵
            t: 3x1 平移向量
        """
        R = pose[:3, :3]
        t = pose[:3, 3]
        return R, t
