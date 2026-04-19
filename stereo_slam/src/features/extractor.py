"""
特征提取模块
使用 ORB 算法从图像中提取特征点和描述子
"""

import cv2
import numpy as np
from typing import Tuple, List


class FeatureExtractor:
    """ORB 特征提取器"""
    
    def __init__(
        self,
        n_features: int = 2000,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        scale_factor: float = 1.2
    ):
        """
        初始化 ORB 特征提取器
        
        Args:
            n_features: 最大特征点数量
            n_levels: 图像金字塔层数
            edge_threshold: 边缘阈值
            first_level: 金字塔第一层
            scale_factor: 金字塔缩放因子
        """
        self.n_features = n_features
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        
        self.orb = cv2.ORB.create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level
        )
        
    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        从图像中提取 ORB 特征
        
        Args:
            image: 输入图像 (灰度图或 BGR 图)
            
        Returns:
            keypoints: 特征点列表
            descriptors: 描述子数组
        """
        # 如果是彩色图像，转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 提取特征
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_stereo(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[
        List[cv2.KeyPoint], List[cv2.KeyPoint], 
        np.ndarray, np.ndarray
    ]:
        """
        从立体图像对中提取特征
        
        Args:
            left_img: 左眼图像
            right_img: 右眼图像
            
        Returns:
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            left_descriptors: 左图描述子
            right_descriptors: 右图描述子
        """
        left_keypoints, left_descriptors = self.extract(left_img)
        right_keypoints, right_descriptors = self.extract(right_img)
        
        return left_keypoints, right_keypoints, left_descriptors, right_descriptors