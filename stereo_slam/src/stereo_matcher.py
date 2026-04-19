"""
立体匹配模块
用于匹配左右图像的特征点
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class StereoMatcher:
    """立体特征匹配器"""
    
    def __init__(
        self,
        ratio_threshold: float = 0.75,
        cross_check: bool = False
    ):
        """
        初始化立体匹配器
        
        Args:
            ratio_threshold: Lowe's ratio 测试阈值
            cross_check: 是否使用交叉验证
        """
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        
        # 创建 BFMatcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        
    def match_stereo_features(
        self,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        left_descriptors: np.ndarray,
        right_descriptors: np.ndarray
    ) -> List[Tuple[cv2.DMatch, cv2.DMatch]]:
        """
        匹配左右图像的特征点
        
        Args:
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            left_descriptors: 左图描述子
            right_descriptors: 右图描述子
            
        Returns:
            匹配对列表，每个元素是 (左图匹配，右图匹配) 的元组
        """
        if left_descriptors is None or right_descriptors is None:
            return []
            
        # 使用 KNN 匹配
        matches = self.matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append((m, n))
                
        return good_matches
    
    def match_stereo_rectified(
        self,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        left_descriptors: np.ndarray,
        right_descriptors: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        对于校正后的立体图像，进行特征匹配
        由于校正后对应点应该在同一水平线上，可以使用更简单的匹配
        
        Args:
            left_keypoints: 左图特征点
            right_keypoints: 右图特征点
            left_descriptors: 左图描述子
            right_descriptors: 右图描述子
            
        Returns:
            匹配对列表
        """
        if left_descriptors is None or right_descriptors is None:
            return []
            
        # 使用 BFMatcher 进行匹配
        matches = self.matcher.match(left_descriptors, right_descriptors)
        
        # 过滤低质量匹配
        if matches:
            distances = [m.distance for m in matches]
            median_distance = np.median(distances)
            
            # 过滤距离过大的匹配
            matches = [m for m in matches if m.distance < 1.5 * median_distance]
            
        return matches