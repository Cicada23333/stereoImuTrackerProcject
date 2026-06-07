"""
3D 点数据结构
改进版：支持加权平均更新和观测管理
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class Point3D:
    """3D 点数据结构"""
    position: np.ndarray  # [x, y, z]
    color: Optional[np.ndarray] = None  # [b, g, r]
    descriptor: Optional[np.ndarray] = None  # ORB descriptor for map localization
    observation_count: int = 0  # 观测次数
    last_seen_frame: int = 0  # 最后看到的帧号
    observation_ids: List[int] = field(default_factory=list)  # 所有观测该点的帧 ID
    
    # 用于加权平均更新的累积值
    _position_sum: np.ndarray = field(default=None)
    _observation_weight: float = 0.0
    
    def __post_init__(self):
        """确保数据是 numpy 数组"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if self.color is not None and not isinstance(self.color, np.ndarray):
            self.color = np.array(self.color, dtype=np.uint8)
        if self.descriptor is not None and not isinstance(self.descriptor, np.ndarray):
            self.descriptor = np.array(self.descriptor, dtype=np.uint8)
        if self.descriptor is not None:
            self.descriptor = self.descriptor.astype(np.uint8, copy=False).reshape(-1)

    def mark_observed(self, frame_id: int):
        """记录一次观测，但不改变 3D 位置。"""
        if frame_id not in self.observation_ids:
            self.observation_ids.append(frame_id)
        self.observation_count += 1
        self.last_seen_frame = frame_id
    
    def add_observation(self, frame_id: int, position: np.ndarray, 
                        weight: float = 1.0, use_weighted_average: bool = True):
        """
        添加观测并更新位置
        
        Args:
            frame_id: 观测帧 ID
            position: 新观测的 3D 位置
            weight: 观测权重
            use_weighted_average: 是否使用加权平均
        """
        previous_count = max(self.observation_count, 1)
        self.mark_observed(frame_id)
        
        if use_weighted_average:
            observation_weight = max(float(weight), 1e-6)
            self.position = (
                self.position * previous_count + position * observation_weight
            ) / (previous_count + observation_weight)
        else:
            # 简单平均
            if self._position_sum is None:
                self._position_sum = self.position.copy() * previous_count
                self._observation_weight = float(previous_count)
            
            self._position_sum += position
            self._observation_weight += 1
            self.position = self._position_sum / self._observation_weight

    def update_descriptor(self, descriptor: Optional[np.ndarray]):
        if descriptor is None:
            return
        self.descriptor = np.array(descriptor, dtype=np.uint8).reshape(-1).copy()
    
    def get_confidence(self) -> float:
        """
        获取点的置信度
        基于观测次数和观测跨度
        """
        if self.observation_count == 0:
            return 0.0
        
        # 观测次数越多，置信度越高（有上限）
        obs_confidence = min(self.observation_count / 10.0, 1.0)
        
        return obs_confidence
    
    def should_cull(self, min_observations: int = 2) -> bool:
        """
        判断是否应该删除该点
        
        Args:
            min_observations: 最小观测次数
        """
        return self.observation_count < min_observations
