"""Pose and coordinate conversion helpers."""

from typing import Optional

import numpy as np


class PoseUtilsMixin:
    """Helpers for world/camera coordinate transforms."""

    def _camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        if points_camera.size == 0:
            return points_camera.copy()

        rotation_world_to_camera = self.camera_pose[:3, :3]
        translation_world_to_camera = self.camera_pose[:3, 3]
        return (
            rotation_world_to_camera.T
            @ (points_camera.T - translation_world_to_camera.reshape(3, 1))
        ).T

    def _camera_center_world(self, camera_pose: Optional[np.ndarray] = None) -> np.ndarray:
        pose = self.camera_pose if camera_pose is None else camera_pose
        rotation_world_to_camera = pose[:3, :3]
        translation_world_to_camera = pose[:3, 3]
        return -rotation_world_to_camera.T @ translation_world_to_camera

    def get_camera_pose(self) -> np.ndarray:
        """Return the current world-to-camera pose."""
        return self.camera_pose.copy()

    def get_camera_position(self) -> np.ndarray:
        """Return the current camera center in world coordinates."""
        return self._camera_center_world().copy()

