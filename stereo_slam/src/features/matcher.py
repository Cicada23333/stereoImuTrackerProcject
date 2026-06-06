"""
特征匹配模块
用于匹配左右图像的特征点
"""

import cv2
import numpy as np
from typing import Tuple, List


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
        right_descriptors: np.ndarray,
        max_vertical_diff: float = 2.0,
        min_disparity: float = 1.0,
        max_disparity: float = 1000.0
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
        if len(left_descriptors) < 2 or len(right_descriptors) < 2:
            return []

        if self.cross_check:
            matches = self.matcher.match(left_descriptors, right_descriptors)
        else:
            raw_matches = self.matcher.knnMatch(left_descriptors, right_descriptors, k=2)
            matches = []
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    matches.append(m)

        filtered_matches = []
        for match in matches:
            left_pt = left_keypoints[match.queryIdx].pt
            right_pt = right_keypoints[match.trainIdx].pt
            disparity = left_pt[0] - right_pt[0]
            vertical_diff = abs(left_pt[1] - right_pt[1])

            if vertical_diff > max_vertical_diff:
                continue
            if disparity < min_disparity or disparity > max_disparity:
                continue
            filtered_matches.append(match)

        filtered_matches.sort(key=lambda m: m.distance)
        return filtered_matches
