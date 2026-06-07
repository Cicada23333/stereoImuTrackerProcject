"""Map persistence and visualization helpers."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from ..map import Map


class MapIOMixin:
    """Map statistics, persistence, and simple visualization."""

    def get_map_statistics(self):
        return self.map.get_statistics()

    def visualize_map(self, save_path: Optional[Union[str, Path]] = None):
        positions = self.map.get_3d_points_array()
        if len(positions) == 0:
            self.logger.warning("No points to visualize")
            return None

        width, height = 800, 600
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)

        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        x_scale = (width - 40) / max(x_max - x_min, 0.1)
        z_scale = (height - 40) / max(z_max - z_min, 0.1)
        colors = self.map.get_3d_points_colors()

        for index, position in enumerate(positions):
            x = int((position[0] - x_min) * x_scale + 20)
            y = int(height - 20 - (position[2] - z_min) * z_scale)
            if not (0 <= x < width and 0 <= y < height):
                continue
            color = tuple(colors[index].tolist()) if colors is not None and index < len(colors) else (255, 255, 255)
            cv2.circle(vis_img, (x, y), 2, color, -1)

        cv2.putText(
            vis_img,
            f"3D Map Visualization - {len(positions)} points",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        for keyframe in self.map.get_all_keyframes():
            if keyframe.camera_pose is None:
                continue
            camera_position = self._camera_center_world(np.array(keyframe.camera_pose))
            x = int((camera_position[0] - x_min) * x_scale + 20)
            y = int(height - 20 - (camera_position[2] - z_min) * z_scale)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis_img, (x, y), 4, (0, 255, 255), -1)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis_img)
            self.logger.info(f"Visualization saved to {save_path}")

        return vis_img

    def save_map(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.map.save_to_file(str(filepath))
        self.logger.info(f"Map saved to {filepath}")

    def load_map(self, filepath: Union[str, Path]):
        self.map = Map.load_from_file(str(filepath))
        self.logger.info(f"Map loaded from {filepath}")

