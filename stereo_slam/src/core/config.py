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
    image_width: int = 2560
    image_height: int = 800
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


@dataclass
class VOConfig:
    """视觉里程计配置"""
    min_inliers_ratio: float = 0.15  # 最小内点比例
    ransac_reproj_threshold: float = 3.0  # RANSAC 重投影阈值
    min_matches: int = 10  # 最小匹配数量


@dataclass
class MapConfig:
    """地图配置"""
    # 点关联配置
    distance_threshold: float = 0.1  # 距离阈值（更严格）
    max_observation_distance: float = 0.3  # 最大观测距离（更严格）
    min_observations: int = 3  # 最小观测次数（需要更多观测才被认为是可靠点）
    
    # 三角测量配置 - 收紧限制以提高点质量
    min_disparity: float = 2.0  # 最小视差（提高以拒绝远距离/噪声点）
    max_disparity: float = 200.0  # 最大视差（降低以拒绝过近/不可靠点）
    min_depth: float = 1.0  # 最小深度（米，提高）
    max_depth: float = 15.0  # 最大深度（米，降低）
    
    # 缓存配置
    max_cache_size: int = 500  # 最大缓存大小（降低）
    
    # 点更新配置 - 使用更保守的权重
    update_weight: float = 0.05  # 新观测的权重（更保守）
    
    # 深度稳定性配置
    depth_variance_threshold: float = 0.05  # 深度方差阈值（更严格）
    min_stereo_baseline: float = 0.02  # 最小立体基线变化（提高）
    
    # 新点质量要求
    min_reprojection_error: float = 0.5  # 最大重投影误差
    min_parallax_angle: float = 5.0  # 最小视差角（度）


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