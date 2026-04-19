"""
主 SLAM 系统模块
整合所有组件实现立体视觉 SLAM
支持增量式地图更新和相机位姿估计
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from .feature_extractor import FeatureExtractor
from .stereo_matcher import StereoMatcher
from .point_cloud import PointCloud, StereoTriangulator, Point3D
from .map import Map, KeyFrame
from .visual_odometry import VisualOdometry


# 配置默认日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class StereoSLAM:
    """立体视觉 SLAM 系统"""
    
    def __init__(
        self,
        device_id: int = 0,
        baseline: float = 0.065,  # 65mm 基线
        focal_length: float = 1000.0,
        image_width: int = 2560,
        image_height: int = 720,
        fov_horizontal: float = 100.0,  # 水平 FOV
        debug_mode: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化立体 SLAM 系统
        
        Args:
            device_id: 设备 ID
            baseline: 左右相机基线距离（米）
            focal_length: 焦距（像素）
            image_width: 图像宽度
            image_height: 图像高度
            fov_horizontal: 水平视场角（度）
            debug_mode: 是否启用调试模式
            logger: 日志记录器
        """
        self.device_id = device_id
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height
        self.debug_mode = debug_mode
        
        # 计算焦距从 FOV
        focal_from_fov = image_width / (2 * np.tan(np.deg2rad(fov_horizontal / 2)))
        self.focal_length = focal_from_fov
        
        # 主点（图像中心）
        self.principal_point = (image_width / 2, image_height / 2)
        
        # 初始化组件
        self.logger = logger or logging.getLogger(f"stereo_slam_{device_id}")
        self.feature_extractor = FeatureExtractor(n_features=2000)
        self.stereo_matcher = StereoMatcher(ratio_threshold=0.75)
        self.triangulator = StereoTriangulator(
            baseline=baseline,
            focal_length=self.focal_length,
            principal_point=self.principal_point
        )
        self.map = Map(device_id=device_id)
        self.point_cloud = PointCloud()
        
        # 相机内参矩阵
        self.K = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 相机位姿（初始为单位矩阵）
        self.camera_pose = np.eye(4)
        
        # 视觉里程计
        self.visual_odometry = VisualOdometry(self.K)
        
        # 用于跟踪的 3D 点缓存
        self.current_3d_points: Optional[np.ndarray] = None
        self.current_left_keypoints: List[cv2.KeyPoint] = []
        self.current_left_descriptors: Optional[np.ndarray] = None
        
        self.logger.info(f"StereoSLAM initialized for device {device_id}")
        self.logger.info(f"  Baseline: {baseline}m")
        self.logger.info(f"  Focal length: {self.focal_length:.2f} pixels")
        self.logger.info(f"  FOV: {fov_horizontal}°")
        self.logger.info(f"  Image size: {image_width}x{image_height}")
        
    def process_frame(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        frame_id: Optional[int] = None
    ) -> Dict:
        """
        处理一帧立体图像
        
        Args:
            left_image: 左眼图像
            right_image: 右眼图像
            frame_id: 帧 ID，如果为 None 则自动递增
            
        Returns:
            处理结果的字典
        """
        if frame_id is None:
            frame_id = self.map.frame_counter
            
        self.logger.debug(f"Processing frame {frame_id}")
        
        # 1. 提取特征
        left_keypoints, right_keypoints, left_descriptors, right_descriptors = \
            self.feature_extractor.extract_stereo(left_image, right_image)
        
        self.logger.info(f"  Extracted {len(left_keypoints)} left and {len(right_keypoints)} right keypoints")
        
        if not left_keypoints or not right_keypoints:
            return {
                "frame_id": frame_id,
                "success": False,
                "error": "No features detected",
                "num_matches": 0
            }
        
        # 2. 立体匹配
        matches = self.stereo_matcher.match_stereo_rectified(
            left_keypoints, right_keypoints,
            left_descriptors, right_descriptors
        )
        
        self.logger.info(f"  Found {len(matches)} stereo matches")
        
        if not matches:
            return {
                "frame_id": frame_id,
                "success": False,
                "error": "No stereo matches found",
                "num_matches": 0
            }
        
        # 3. 三角测量新的 3D 点
        triangulated_points = self.triangulator.triangulate_matches(
            left_keypoints, right_keypoints, matches
        )
        
        self.logger.info(f"  Triangulated {len(triangulated_points)} 3D points")
        
        # 4. 使用视觉里程计估计相机位姿（如果已有地图点）
        num_matches_with_map = 0
        num_inliers = 0
        if frame_id > 0 and self.current_3d_points is not None:
            pose, num_matches_with_map, num_inliers = self.visual_odometry.update(
                left_keypoints, left_descriptors
            )
            self.camera_pose = pose
            self.logger.info(f"  VO: {num_inliers} inliers from {num_matches_with_map} matches")
        
        # 5. 更新地图 - 添加新点或更新已有点
        new_points_count = 0
        updated_points_count = 0
        
        for feature_id, position in triangulated_points:
            left_pt = left_keypoints[feature_id].pt
            color = None
            if 0 <= int(left_pt[1]) < left_image.shape[0] and 0 <= int(left_pt[0]) < left_image.shape[1]:
                color = left_image[int(left_pt[1]), int(left_pt[0])]
            
            # 检查是否已有相近的 3D 点
            existing_point_id = self._find_nearby_point(position)
            
            if existing_point_id is not None:
                self.map.update_3d_point(
                    existing_point_id,
                    position=position,
                    color=color,
                    add_observation=frame_id
                )
                updated_points_count += 1
            else:
                point_id = self.map.add_3d_point(
                    position=position,
                    color=color,
                    observation_ids=[frame_id]
                )
                self.point_cloud.add_point(position, color, point_id)
                new_points_count += 1
        
        self.logger.info(f"  Added {new_points_count} new points, updated {updated_points_count} existing points")
        
        # 6. 添加关键帧
        keyframe = self.map.add_keyframe(
            frame_id=frame_id,
            left_image=left_image,
            right_image=right_image,
            left_keypoints=left_keypoints,
            right_keypoints=right_keypoints,
            left_descriptors=left_descriptors,
            right_descriptors=right_descriptors,
            camera_pose=self.camera_pose.copy()
        )
        
        # 7. 更新视觉里程计的缓存
        self._update_vo_cache(left_keypoints, left_descriptors, triangulated_points)
        
        # 返回结果
        result = {
            "frame_id": frame_id,
            "success": True,
            "num_keypoints_left": len(left_keypoints),
            "num_keypoints_right": len(right_keypoints),
            "num_matches": len(matches),
            "num_new_points": new_points_count,
            "num_updated_points": updated_points_count,
            "total_map_points": len(self.map.points),
            "camera_pose": self.camera_pose.tolist(),
            "vo_matches": num_matches_with_map,
            "vo_inliers": num_inliers,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.debug_mode:
            result["debug_info"] = self._get_debug_info(left_image, right_image, 
                                                         left_keypoints, matches)
        
        return result
    
    def _find_nearby_point(self, position: np.ndarray, threshold: float = 0.05) -> Optional[int]:
        """查找附近已存在的 3D 点
        
        使用更严格的阈值，只有非常接近的点才被认为是同一个点。
        这样可以确保新视角下的新特征能够被添加到地图中。
        """
        if not self.map.points:
            return None
        
        closest_id = None
        closest_distance = float('inf')
        
        for point_id, point_data in self.map.points.items():
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
    
    def _update_vo_cache(
        self, 
        keypoints: List[cv2.KeyPoint], 
        descriptors: np.ndarray,
        triangulated_points: List[Tuple[int, np.ndarray]]
    ):
        """更新视觉里程计的缓存数据"""
        if not triangulated_points:
            return
        
        # 构建当前帧的 3D 点数组
        positions = np.array([p[1] for p in triangulated_points], dtype=np.float32)
        valid_indices = [p[0] for p in triangulated_points]
        
        # 限制缓存大小
        max_cache_size = 500
        if len(positions) > max_cache_size:
            indices = np.random.choice(len(positions), max_cache_size, replace=False)
            positions = positions[indices]
            valid_indices = [valid_indices[i] for i in indices]
        
        self.current_3d_points = positions
        self.current_left_keypoints = [keypoints[i] for i in valid_indices if i < len(keypoints)]
        self.current_left_descriptors = descriptors[valid_indices] if descriptors is not None else None
        
        # 初始化或更新视觉里程计
        if len(self.current_3d_points) > 10:
            if self.visual_odometry.prev_3d_points is None:
                self.visual_odometry.initialize(
                    self.current_left_keypoints,
                    self.current_left_descriptors,
                    self.current_3d_points
                )
            else:
                self.visual_odometry.prev_keypoints = self.current_left_keypoints
                self.visual_odometry.prev_descriptors = self.current_left_descriptors
                self.visual_odometry.prev_3d_points = self.current_3d_points
    
    def _get_debug_info(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Dict:
        """获取调试信息"""
        debug_img = left_image.copy()
        
        for kp in keypoints[:100]:
            cv2.circle(debug_img, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 0), -1)
        
        for match in matches[:50]:
            pt1 = (int(keypoints[match.queryIdx].pt[0]), 
                   int(keypoints[match.queryIdx].pt[1]))
            cv2.circle(debug_img, pt1, 5, (255, 0, 0), -1)
        
        return {
            "debug_image_shape": debug_img.shape,
            "matched_points_displayed": min(50, len(matches))
        }
    
    def get_map_statistics(self) -> Dict:
        """获取地图统计信息"""
        return self.map.get_statistics()
    
    def get_camera_pose(self) -> np.ndarray:
        """获取当前相机位姿"""
        return self.camera_pose.copy()
    
    def get_camera_position(self) -> np.ndarray:
        """获取当前相机位置"""
        return self.camera_pose[:3, 3].copy()
    
    def visualize_map(self, save_path: Optional[str] = None):
        """可视化 3D 地图"""
        positions = self.map.get_3d_points_array()
        
        if len(positions) == 0:
            self.logger.warning("No points to visualize")
            return
        
        width, height = 800, 600
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(positions) > 0:
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
            
            x_scale = (width - 40) / max(x_max - x_min, 0.1)
            y_scale = (height - 40) / max(z_max - z_min, 0.1)
            
            colors = self.map.get_3d_points_colors()
            
            for i, pos in enumerate(positions):
                x = int((pos[0] - x_min) * x_scale + 20)
                y = int(height - 20 - (pos[2] - z_min) * y_scale)
                
                if 0 <= x < width and 0 <= y < height:
                    if colors is not None and i < len(colors):
                        color = tuple(colors[i].tolist())
                    else:
                        color = (255, 255, 255)
                    
                    cv2.circle(vis_img, (x, y), 2, color, -1)
        
        cv2.putText(vis_img, f"3D Map Visualization - {len(positions)} points", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制相机轨迹
        keyframes = self.map.get_all_keyframes()
        if keyframes:
            for kf in keyframes:
                if kf.camera_pose is not None:
                    pose = np.array(kf.camera_pose)
                    cam_pos = pose[:3, 3]
                    
                    x = int((cam_pos[0] - x_min) * x_scale + 20)
                    y = int(height - 20 - (cam_pos[2] - z_min) * y_scale)
                    
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(vis_img, (x, y), 4, (0, 255, 255), -1)
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
            self.logger.info(f"Visualization saved to {save_path}")
        
        return vis_img
    
    def save_map(self, filepath: str):
        """保存地图到文件"""
        self.map.save_to_file(filepath)
        self.logger.info(f"Map saved to {filepath}")
    
    def load_map(self, filepath: str):
        """从文件加载地图"""
        self.map = Map.load_from_file(filepath)
        self.logger.info(f"Map loaded from {filepath}")