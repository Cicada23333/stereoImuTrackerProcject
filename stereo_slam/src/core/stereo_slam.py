"""
主 SLAM 系统模块
整合所有组件实现立体视觉 SLAM
改进版：使用观测驱动的地图更新和加权平均策略
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import logging
from datetime import datetime

from ..features import FeatureExtractor, StereoMatcher
from ..geometry import StereoTriangulator, GeometryUtils
from ..map import Map, KeyFrame
from ..vo import VisualOdometry
from .config import SLAMConfig, CameraConfig


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
        focal_length: Optional[float] = None,
        image_width: int = 1280,
        image_height: int = 720,
        stereo_width: Optional[int] = None,
        fov_horizontal: float = 100.0,  # 水平 FOV
        eye_order: str = "left-right",
        auto_save_path: Optional[Union[str, Path]] = None,
        debug_mode: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化立体 SLAM 系统
        
        Args:
            device_id: 设备 ID
            baseline: 左右相机基线距离（米）
            focal_length: 焦距（像素）。为 None 时从 image_width 和 fov_horizontal 计算
            image_width: 单只眼图像宽度。2560x720 拼接图应使用默认 1280
            image_height: 图像高度
            stereo_width: 左右拼接图总宽度。默认是 image_width * 2
            fov_horizontal: 水平视场角（度）
            eye_order: 拼接顺序，"left-right" 或 "right-left"
            auto_save_path: 每次高层处理完成后自动保存地图的位置
            debug_mode: 是否启用调试模式
            logger: 日志记录器
        """
        self.device_id = device_id
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height
        self.stereo_width = stereo_width or image_width * 2
        if eye_order not in ("left-right", "right-left"):
            raise ValueError('eye_order 必须是 "left-right" 或 "right-left"')
        self.eye_order = eye_order
        self.auto_save_path = Path(auto_save_path) if auto_save_path else None
        self.debug_mode = debug_mode
        
        # 计算焦距从 FOV
        focal_from_fov = image_width / (2 * np.tan(np.deg2rad(fov_horizontal / 2)))
        self.focal_length = float(focal_length) if focal_length is not None else float(focal_from_fov)
        
        # 主点（图像中心）
        self.principal_point = (image_width / 2, image_height / 2)
        
        # 初始化配置
        self.config = SLAMConfig(
            device_id=device_id,
            debug_mode=debug_mode,
            camera=CameraConfig(
                image_width=image_width,
                image_height=image_height,
                fov_horizontal=fov_horizontal,
                baseline=baseline,
                focal_length=self.focal_length
            )
        )
        
        # 初始化组件
        self.logger = logger or logging.getLogger(f"stereo_slam_{device_id}")
        self.feature_extractor = FeatureExtractor(n_features=2000)
        self.stereo_matcher = StereoMatcher(ratio_threshold=0.75)
        self.triangulator = StereoTriangulator(
            baseline=baseline,
            focal_length=self.focal_length,
            principal_point=self.principal_point
        )
        # 使用配置初始化地图
        self.map = Map(
            device_id=device_id,
            min_observations=self.config.map.min_observations
        )
        
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
        self.last_observations: List[Dict] = []
        
        self.logger.info(f"StereoSLAM initialized for device {device_id}")
        self.logger.info(f"  Baseline: {baseline}m")
        self.logger.info(f"  Focal length: {self.focal_length:.2f} pixels")
        self.logger.info(f"  FOV: {fov_horizontal}°")
        self.logger.info(f"  Eye image size: {image_width}x{image_height}")
        self.logger.info(f"  Side-by-side image size: {self.stereo_width}x{image_height}")
        self.logger.info(f"  Eye order: {self.eye_order}")

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """加载用户传入的图像。支持 numpy 数组和图片路径。"""
        if isinstance(image, np.ndarray):
            return image

        image_path = Path(image)
        loaded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return loaded

    def split_stereo_image(self, stereo_image: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将横向拼接双目图拆成左右眼图像。

        默认期望输入是 2560x720，拆分后每只眼是 1280x720。
        """
        image = self._load_image(stereo_image)
        if image.ndim not in (2, 3):
            raise ValueError(f"图像必须是灰度或 BGR/BGRA 数组，当前维度: {image.ndim}")

        height, width = image.shape[:2]
        if width != self.stereo_width or height != self.image_height:
            raise ValueError(
                f"双目拼接图尺寸不匹配，期望 {self.stereo_width}x{self.image_height}，"
                f"实际 {width}x{height}。如果相机输出不同，请用对应 image_width/image_height 初始化。"
            )

        if width % 2 != 0:
            raise ValueError(f"双目拼接图宽度必须是偶数，实际宽度: {width}")

        mid = width // 2
        first_half = image[:, :mid]
        second_half = image[:, mid:]
        if self.eye_order == "left-right":
            left_image = first_half.copy()
            right_image = second_half.copy()
        else:
            left_image = second_half.copy()
            right_image = first_half.copy()
        return left_image, right_image

    def process_stereo_image(
        self,
        stereo_image: Union[str, Path, np.ndarray],
        frame_id: Optional[int] = None,
        save_map_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        处理一张横向拼接的双目图，并增量更新地图。

        Args:
            stereo_image: 2560x720 左右拼接图，或图片路径
            frame_id: 帧 ID。如果为 None，则自动递增
            save_map_path: 处理后保存地图的 JSON 路径
        """
        input_image = self._load_image(stereo_image)
        left_image, right_image = self.split_stereo_image(input_image)
        result = self.process_frame(left_image, right_image, frame_id=frame_id)
        result["input_shape"] = list(input_image.shape)
        result["left_shape"] = list(left_image.shape)
        result["right_shape"] = list(right_image.shape)
        result["eye_order"] = self.eye_order
        self._save_map_if_requested(save_map_path)
        return result

    def process_images(
        self,
        images: Iterable[Union[str, Path, np.ndarray]],
        save_map_path: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """按顺序处理多张横向拼接双目图，持续更新同一张地图。"""
        results = []
        for image in images:
            results.append(self.process_stereo_image(image, save_map_path=None))

        self._save_map_if_requested(save_map_path)
        return results

    def process(
        self,
        images: Union[str, Path, np.ndarray, Iterable[Union[str, Path, np.ndarray]]],
        save_map_path: Optional[Union[str, Path]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        统一入口：传单张图返回一个结果，传多张图返回结果列表。
        """
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return self.process_images(list(images), save_map_path=save_map_path)
        if isinstance(images, (str, Path, np.ndarray)):
            return self.process_stereo_image(images, save_map_path=save_map_path)
        return self.process_images(images, save_map_path=save_map_path)

    def _save_map_if_requested(self, save_map_path: Optional[Union[str, Path]] = None):
        target = Path(save_map_path) if save_map_path else self.auto_save_path
        if target is not None:
            self.save_map(target)

    def _camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """将当前相机坐标点转换为世界/地图坐标。"""
        if points_camera.size == 0:
            return points_camera.copy()

        R_cw = self.camera_pose[:3, :3]
        t_cw = self.camera_pose[:3, 3]
        return ((R_cw.T @ (points_camera.T - t_cw.reshape(3, 1))).T)

    def _camera_center_world(self, camera_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """从 world->camera 位姿计算相机中心的世界坐标。"""
        pose = self.camera_pose if camera_pose is None else camera_pose
        R_cw = pose[:3, :3]
        t_cw = pose[:3, 3]
        return -R_cw.T @ t_cw
        
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
        if left_image.shape[:2] != right_image.shape[:2]:
            raise ValueError(
                f"左右眼图像尺寸必须一致，left={left_image.shape[:2]}, right={right_image.shape[:2]}"
            )
        if left_image.shape[1] != self.image_width or left_image.shape[0] != self.image_height:
            raise ValueError(
                f"单眼图像尺寸不匹配，期望 {self.image_width}x{self.image_height}，"
                f"实际 {left_image.shape[1]}x{left_image.shape[0]}"
            )

        if frame_id is None:
            frame_id = self.map.frame_counter
        self.map.frame_counter = max(self.map.frame_counter, frame_id + 1)
        self.last_observations = []
            
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
            left_descriptors, right_descriptors,
            max_vertical_diff=self.config.map.max_vertical_disparity,
            min_disparity=self.config.map.min_disparity,
            max_disparity=self.config.map.max_disparity
        )
        
        self.logger.info(f"  Found {len(matches)} stereo matches")
        
        if not matches:
            return {
                "frame_id": frame_id,
                "success": False,
                "error": "No stereo matches found",
                "num_matches": 0
            }
        
        # 3. 三角测量新的 3D 点（带质量检查）
        triangulated_points = self._triangulate_with_quality_check(
            left_keypoints, right_keypoints, matches
        )
        
        self.logger.info(f"  Triangulated {len(triangulated_points)} valid 3D points")
        
        # 4. 使用视觉里程计估计相机位姿（如果已有地图点）
        num_matches_with_map = 0
        num_inliers = 0
        camera_moved_significant = False  # 相机是否移动足够添加关键帧
        camera_movement_distance = 0.0  # 初始化相机移动距离
        
        if frame_id > 0 and self.current_3d_points is not None:
            pose, num_matches_with_map, num_inliers = self.visual_odometry.update(
                left_keypoints, left_descriptors
            )
            self.camera_pose = pose
            
            # 计算相机移动距离和角度
            current_pos = self._camera_center_world()
            if hasattr(self, '_prev_camera_pos'):
                camera_movement_distance = np.linalg.norm(current_pos - self._prev_camera_pos)
                # 检查是否移动足够添加关键帧
                if camera_movement_distance > self.config.map.min_keyframe_distance:
                    camera_moved_significant = True
            self._prev_camera_pos = current_pos.copy()
            
            self.logger.info(f"  VO: {num_inliers} inliers from {num_matches_with_map} matches, "
                           f"camera moved {camera_movement_distance:.3f}m")

        if triangulated_points:
            feature_ids = [feature_id for feature_id, _ in triangulated_points]
            camera_positions = np.array([position for _, position in triangulated_points], dtype=np.float64)
            world_positions = self._camera_to_world(camera_positions)
            triangulated_world_points = list(zip(feature_ids, world_positions))
        else:
            triangulated_world_points = []
        
        # 5. 更新地图 - 使用 2D 投影关联
        new_points_count = 0
        updated_points_count = 0
        
        # 首先将现有地图点投影到当前帧（用于 2D 关联）
        projected_2d_points = {}  # 映射：(projected_x, projected_y) -> point_id
        if len(self.map.points) > 0:
            camera_pose = self.camera_pose
            K = self.K.copy()
            # 投影所有 3D 点到当前帧
            point_ids = list(self.map.points.keys())
            map_positions = np.array([p.position for p in self.map.points.values()])
            if len(map_positions) > 0:
                projected, projected_indices = GeometryUtils.project_3d_to_2d_with_depth(
                    map_positions, camera_pose, K, left_image.shape, return_indices=True
                )
                # 建立投影点索引
                for pt_2d, point_index in zip(projected, projected_indices):
                    key = (int(pt_2d[0]), int(pt_2d[1]))
                    if key not in projected_2d_points:
                        projected_2d_points[key] = []
                    point_id = point_ids[int(point_index)]
                    projected_2d_points[key].append(point_id)
        
        # 用于 2D 关联的像素搜索半径
        search_radius = int(self.config.map.max_reprojection_pixel_error)
        
        for feature_id, position in triangulated_world_points:
            left_pt = left_keypoints[feature_id].pt
            feature_x, feature_y = int(left_pt[0]), int(left_pt[1])
            color = None
            if 0 <= int(left_pt[1]) < left_image.shape[0] and 0 <= int(left_pt[0]) < left_image.shape[1]:
                color = left_image[int(left_pt[1]), int(left_pt[0])]
            
            # 使用 2D 投影关联检查是否有匹配的地图点
            existing_point_id = self._find_matching_map_point_2d(
                feature_x, feature_y, projected_2d_points, 
                self.map.points, position, search_radius
            )
            
            if existing_point_id is not None:
                # 找到匹配的地图点
                if camera_moved_significant:
                    # 相机移动了，可以更新点位置（使用观测加权平均）
                    self.map.update_3d_point(
                        existing_point_id,
                        position=position,
                        color=color,
                        add_observation=frame_id,
                        use_weighted_average=True,
                        update_weight=self.config.map.update_weight
                    )
                    updated_points_count += 1
                else:
                    # 相机没有明显移动，只增加观测次数，不更新位置
                    self.map.points[existing_point_id].mark_observed(frame_id)
                    updated_points_count += 1
            else:
                # 添加新点
                point_id = self.map.add_3d_point(
                    position=position,
                    color=color,
                    observation_ids=[frame_id]
                )
                new_points_count += 1
        
        self.logger.info(f"  Added {new_points_count} new points, updated {updated_points_count} existing points")
        
        # 6. 只在相机移动足够时才添加关键帧
        if camera_moved_significant or frame_id == 0:
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
            self.logger.info(f"  Added keyframe at frame {frame_id}")
        else:
            keyframe = None
            self.logger.debug(f"  Skipped keyframe (camera not moved enough)")
        
        # 7. 更新视觉里程计的缓存
        self._update_vo_cache(left_keypoints, left_descriptors, triangulated_points)
        
        # 8. 定期清理不可靠的点
        if frame_id > 0 and frame_id % 50 == 0:
            culled_count = self.map.cull_insecure_points()
            if culled_count > 0:
                self.logger.info(f"  Culled {culled_count} insecure points")
        
        # 返回结果
        result = {
            "frame_id": frame_id,
            "success": True,
            "num_keypoints_left": len(left_keypoints),
            "num_keypoints_right": len(right_keypoints),
            "num_matches": len(matches),
            "num_triangulated_points": len(triangulated_points),
            "num_current_observations": len(self.last_observations),
            "num_new_points": new_points_count,
            "num_updated_points": updated_points_count,
            "total_map_points": len(self.map.points),
            "camera_pose": self.camera_pose.tolist(),
            "vo_matches": num_matches_with_map,
            "vo_inliers": num_inliers,
            "camera_moved_significant": camera_moved_significant,
            "camera_movement_distance": camera_movement_distance,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.debug_mode:
            result["debug_info"] = self._get_debug_info(left_image, right_image, 
                                                         left_keypoints, matches)
        
        return result
    
    def _find_matching_map_point_2d(
        self,
        feature_x: int,
        feature_y: int,
        projected_2d_points: Dict,
        map_points: Dict,
        position_3d: np.ndarray,
        search_radius: int = 3
    ) -> Optional[int]:
        """
        使用 2D 投影查找匹配的地图点
        
        Args:
            feature_x: 特征点 x 坐标
            feature_y: 特征点 y 坐标
            projected_2d_points: 投影的 2D 点索引 {(x, y): [point_ids]}
            map_points: 地图点字典 {point_id: Point3D}
            position_3d: 新三角测量的 3D 位置
            search_radius: 搜索半径（像素）
            
        Returns:
            匹配的地图点 ID，如果没有则返回 None
        """
        # 在搜索半径内查找
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                key = (feature_x + dx, feature_y + dy)
                if key in projected_2d_points:
                    # 找到附近的投影点，检查 3D 位置是否接近
                    for point_id in projected_2d_points[key]:
                        if point_id in map_points:
                            existing_pos = np.array(map_points[point_id].position)
                            distance = np.linalg.norm(position_3d - existing_pos)
                            # 如果 3D 距离在可接受范围内，认为是同一个点
                            if distance < self.config.map.distance_threshold * 3:  # 放宽一些
                                return point_id
        return None
    
    def _triangulate_with_quality_check(
        self,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        带质量检查的三角测量
        过滤掉不可靠的三角测量结果
        
        Args:
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            matches: 匹配对列表
            
        Returns:
            高质量的 (feature_id, 3D_position) 列表
        """
        results = []
        self.last_observations = []
        
        for match in matches:
            left_pt = left_keypoints[match.queryIdx].pt
            right_pt = right_keypoints[match.trainIdx].pt
            
            # 计算视差
            disparity = left_pt[0] - right_pt[0]
            vertical_disparity = abs(left_pt[1] - right_pt[1])
            
            # 检查视差范围
            if vertical_disparity > self.config.map.max_vertical_disparity:
                continue
            if disparity < self.config.map.min_disparity:
                continue
            if disparity > self.config.map.max_disparity:
                continue
            
            # 三角测量
            position = self.triangulator.triangulate_point(left_pt, right_pt)
            
            if position is None:
                continue
            
            # 检查深度范围
            depth = np.linalg.norm(position)
            if depth < self.config.map.min_depth:
                continue
            if depth > self.config.map.max_depth:
                continue
            
            # 通过质量检查
            feature_id = int(match.queryIdx)
            results.append((feature_id, position))
            self.last_observations.append({
                "left": (float(left_pt[0]), float(left_pt[1])),
                "right": (float(right_pt[0]), float(right_pt[1])),
                "depth": float(position[2]),
                "disparity": float(disparity),
                "feature_id": feature_id,
                "right_feature_id": int(match.trainIdx)
            })
        
        return results
    
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
        max_cache_size = self.config.map.max_cache_size
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
        return self._camera_center_world().copy()

    def get_last_observations(self) -> List[Dict]:
        """获取上一帧成功三角化的当前图像观测。"""
        return list(self.last_observations)
    
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
                    cam_pos = self._camera_center_world(pose)
                    
                    x = int((cam_pos[0] - x_min) * x_scale + 20)
                    y = int(height - 20 - (cam_pos[2] - z_min) * y_scale)
                    
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(vis_img, (x, y), 4, (0, 255, 255), -1)
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
            self.logger.info(f"Visualization saved to {save_path}")
        
        return vis_img
    
    def save_map(self, filepath: Union[str, Path]):
        """保存地图到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.map.save_to_file(str(filepath))
        self.logger.info(f"Map saved to {filepath}")
    
    def load_map(self, filepath: Union[str, Path]):
        """从文件加载地图"""
        self.map = Map.load_from_file(str(filepath))
        self.logger.info(f"Map loaded from {filepath}")
