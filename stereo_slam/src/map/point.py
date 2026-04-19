"""
3D 点数据结构
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class Point3D:
    """3D 点数据结构"""
    position: np.ndarray  # [x, y, z]
    color: Optional[np.ndarray] = None  # [b, g, r]
    observation_count: int = 0  # 观测次数
    last_seen_frame: int = 0  # 最后看到的帧号
    
    def __post_init__(self):
        """确保数据是 numpy 数组"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if self.color is not None and not isinstance(self.color, np.ndarray):
            self.color = np.array(self.color, dtype=np.uint8)