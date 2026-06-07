"""Current-frame stereo observation and triangulation helpers."""

from typing import Dict, List, Tuple

import cv2
import numpy as np


class ObservationMixin:
    """Build current-frame triangulated observations."""

    def _triangulate_with_quality_check(
        self,
        left_keypoints: List[cv2.KeyPoint],
        right_keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> List[Tuple[int, np.ndarray]]:
        results = []
        self.last_observations = []

        for match in matches:
            left_pt = left_keypoints[match.queryIdx].pt
            right_pt = right_keypoints[match.trainIdx].pt
            disparity = left_pt[0] - right_pt[0]
            vertical_disparity = abs(left_pt[1] - right_pt[1])

            if vertical_disparity > self.config.map.max_vertical_disparity:
                continue
            if disparity < self.config.map.min_disparity:
                continue
            if disparity > self.config.map.max_disparity:
                continue

            position = self.triangulator.triangulate_point(left_pt, right_pt)
            if position is None:
                continue

            depth = np.linalg.norm(position)
            if depth < self.config.map.min_depth or depth > self.config.map.max_depth:
                continue

            feature_id = int(match.queryIdx)
            results.append((feature_id, position))
            self.last_observations.append(
                {
                    "left": (float(left_pt[0]), float(left_pt[1])),
                    "right": (float(right_pt[0]), float(right_pt[1])),
                    "depth": float(position[2]),
                    "disparity": float(disparity),
                    "feature_id": feature_id,
                    "right_feature_id": int(match.trainIdx),
                }
            )

        return results

    def get_last_observations(self) -> List[Dict]:
        """Return successful current-frame stereo observations."""
        return list(self.last_observations)

