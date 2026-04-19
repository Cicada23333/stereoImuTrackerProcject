"""
Stereo SLAM 库
使用 OpenCV 的 ORB 算法和立体视觉进行 3D 地图构建
"""

from .feature_extractor import FeatureExtractor
from .stereo_matcher import StereoMatcher
from .point_cloud import PointCloud, StereoTriangulator, Point3D
from .map import Map, KeyFrame
from .stereo_slam import StereoSLAM

__version__ = "1.0.0"

__all__ = [
    "FeatureExtractor",
    "StereoMatcher",
    "PointCloud",
    "StereoTriangulator",
    "Point3D",
    "Map",
    "KeyFrame",
    "StereoSLAM"
]