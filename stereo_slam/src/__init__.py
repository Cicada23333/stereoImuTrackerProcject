"""
Stereo SLAM 库
使用 OpenCV 的 ORB 算法和立体视觉进行 3D 地图构建

重构后的模块化结构：
- core: 核心模块 (SLAM 主系统、配置)
- features: 特征处理模块 (提取、匹配、跟踪)
- geometry: 几何计算模块 (三角测量、位姿估计)
- map: 地图模块 (3D 点、关键帧、地图管理)
- vo: 视觉里程计模块 (位姿估计、地图更新)
"""

# 核心模块
from .core.stereo_slam import StereoSLAM
from .core.config import SLAMConfig, CameraConfig, FeatureConfig, MatchingConfig, VOConfig, MapConfig

# 特征处理模块
from .features import FeatureExtractor, StereoMatcher, FeatureTracker

# 几何计算模块
from .geometry import StereoTriangulator, PoseEstimator, GeometryUtils

# 地图模块
from .map import Map, KeyFrame, Point3D

# 视觉里程计模块
from .vo import VisualOdometry, MapUpdater

__version__ = "2.0.0"

__all__ = [
    # 核心
    "StereoSLAM",
    "SLAMConfig",
    "CameraConfig",
    "FeatureConfig",
    "MatchingConfig",
    "VOConfig",
    "MapConfig",
    # 特征处理
    "FeatureExtractor",
    "StereoMatcher",
    "FeatureTracker",
    # 几何计算
    "StereoTriangulator",
    "PoseEstimator",
    "GeometryUtils",
    # 地图
    "Map",
    "KeyFrame",
    "Point3D",
    # 视觉里程计
    "VisualOdometry",
    "MapUpdater",
]