"""
SLAM 配置模块
包含 SLAM 系统的所有配置参数
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraConfig:
    """相机配置"""
    image_width: int = 1280
    image_height: int = 720
    fov_horizontal: float = 100.0  # 水平视场角（度）
    baseline: float = 0.065  # 基线距离（米）
    focal_length: float = 1000.0  # 焦距（像素）
    
    # 主点（图像中心）
    principal_point: Optional[tuple] = None
    
    def __post_init__(self):
        """计算默认主点"""
        if self.principal_point is None:
            self.principal_point = (self.image_width / 2, self.image_height / 2)
    
    def get_intrinsics(self) -> np.ndarray:
        """获取相机内参矩阵"""
        cx, cy = self.principal_point
        return np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class FeatureConfig:
    """特征提取配置"""
    n_features: int = 2000  # 最大特征点数量
    n_levels: int = 8  # 图像金字塔层数
    edge_threshold: int = 31  # 边缘阈值
    first_level: int = 0  # 金字塔第一层
    scale_factor: float = 1.2  # 金字塔缩放因子


@dataclass
class MatchingConfig:
    """特征匹配配置"""
    ratio_threshold: float = 0.75  # Lowe's ratio 测试阈值
    cross_check: bool = False  # 是否使用交叉验证
    max_vertical_diff: float = 20.0  # 左右匹配点的最大 y 方向偏差


@dataclass
class VOConfig:
    """视觉里程计配置"""
    min_inliers_ratio: float = 0.15  # 最小内点比例
    ransac_reproj_threshold: float = 3.0  # RANSAC 重投影阈值
    min_matches: int = 10  # 最小匹配数量


@dataclass
class MapConfig:
    """地图配置 - 优化后的抗漂移配置"""
    # 点关联配置 - 使用更严格的阈值
    distance_threshold: float = 0.05  # 距离阈值 (5cm, 防止不同物体合并)
    max_observation_distance: float = 0.15  # 最大观测距离 (15cm)
    min_observations: int = 3  # 最小观测次数 (需要更多观测才被认为是可靠点)
    
    # 三角测量配置 - 更严格的限制以提高点质量
    min_disparity: float = 2.0  # 最小视差 (像素)
    max_disparity: float = 300.0  # 最大视差 (允许室内近距离特征)
    max_vertical_disparity: float = 20.0  # 左右匹配点最大垂直偏差
    min_depth: float = 0.25  # 最小深度 (米)
    max_depth: float = 12.0  # 最大深度 (12 米，超出此距离基线不够可靠)
    
    # 缓存配置
    max_cache_size: int = 500  # 最大缓存大小
    
    # 点更新配置 - 使用观测加权平均
    update_weight: float = 0.05  # 新观测的权重 (更保守)
    
    # 深度稳定性配置
    depth_variance_threshold: float = 0.05  # 深度方差阈值
    min_stereo_baseline: float = 0.02  # 最小立体基线变化
    
    # 新点质量要求
    min_reprojection_error: float = 0.5  # 最大重投影误差
    min_parallax_angle: float = 5.0  # 最小视差角 (度)
    
    # 关键帧添加条件
    min_keyframe_distance: float = 0.1  # 最小相机移动距离 (10cm) 才添加关键帧
    min_keyframe_angle: float = 10.0  # 最小旋转角度 (度) 才添加关键帧
    
    # 2D 投影关联配置
    max_reprojection_pixel_error: float = 3.0  # 最大重投影像素误差 (用于 2D 关联)


@dataclass
class SLAMConfig:
    """SLAM 系统配置"""
    # 设备配置
    device_id: int = 0
    debug_mode: bool = False
    
    # 子配置
    camera: CameraConfig = field(default_factory=CameraConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    vo: VOConfig = field(default_factory=VOConfig)
    map: MapConfig = field(default_factory=MapConfig)
    
    def get_focal_length_from_fov(self) -> float:
        """从 FOV 计算焦距"""
        return self.camera.image_width / (2 * np.tan(np.deg2rad(self.camera.fov_horizontal / 2)))
