"""Map point association and update helpers."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..geometry import GeometryUtils


class MapAssociationMixin:
    """Associate current observations with existing map points."""

    def _to_world_triangulated_points(
        self,
        triangulated_points: List[Tuple[int, np.ndarray]],
    ) -> List[Tuple[int, np.ndarray]]:
        if not triangulated_points:
            return []
        feature_ids = [feature_id for feature_id, _ in triangulated_points]
        camera_positions = np.array([position for _, position in triangulated_points])
        world_positions = self._camera_to_world(camera_positions)
        return list(zip(feature_ids, world_positions))

    def _update_map_points(
        self,
        triangulated_world_points: List[Tuple[int, np.ndarray]],
        left_keypoints: List[cv2.KeyPoint],
        left_image: np.ndarray,
        frame_id: int,
        camera_moved_significant: bool,
    ) -> Tuple[int, int]:
        projected_2d_points = self._build_projected_point_index(left_image.shape)
        search_radius = int(self.config.map.max_reprojection_pixel_error)
        new_points_count = 0
        updated_points_count = 0

        for feature_id, position in triangulated_world_points:
            left_pt = left_keypoints[feature_id].pt
            feature_x, feature_y = int(left_pt[0]), int(left_pt[1])
            color = None
            if 0 <= feature_y < left_image.shape[0] and 0 <= feature_x < left_image.shape[1]:
                color = left_image[feature_y, feature_x]

            existing_point_id = self._find_matching_map_point_2d(
                feature_x,
                feature_y,
                projected_2d_points,
                self.map.points,
                position,
                search_radius,
            )

            if existing_point_id is not None:
                if camera_moved_significant:
                    self.map.update_3d_point(
                        existing_point_id,
                        position=position,
                        color=color,
                        add_observation=frame_id,
                        use_weighted_average=True,
                        update_weight=self.config.map.update_weight,
                    )
                else:
                    self.map.points[existing_point_id].mark_observed(frame_id)
                updated_points_count += 1
            else:
                self.map.add_3d_point(
                    position=position,
                    color=color,
                    observation_ids=[frame_id],
                )
                new_points_count += 1

        self.logger.info(
            f"  Added {new_points_count} new points, updated {updated_points_count} existing points"
        )
        return new_points_count, updated_points_count

    def _build_projected_point_index(self, image_shape) -> Dict:
        projected_2d_points = {}
        if len(self.map.points) == 0:
            return projected_2d_points

        point_ids = list(self.map.points.keys())
        map_positions = np.array([p.position for p in self.map.points.values()])
        projected, projected_indices = GeometryUtils.project_3d_to_2d_with_depth(
            map_positions,
            self.camera_pose,
            self.K,
            image_shape,
            return_indices=True,
        )

        for pt_2d, point_index in zip(projected, projected_indices):
            key = (int(pt_2d[0]), int(pt_2d[1]))
            projected_2d_points.setdefault(key, []).append(point_ids[int(point_index)])

        return projected_2d_points

    def _find_matching_map_point_2d(
        self,
        feature_x: int,
        feature_y: int,
        projected_2d_points: Dict,
        map_points: Dict,
        position_3d: np.ndarray,
        search_radius: int = 3,
    ) -> Optional[int]:
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                key = (feature_x + dx, feature_y + dy)
                if key not in projected_2d_points:
                    continue
                for point_id in projected_2d_points[key]:
                    if point_id not in map_points:
                        continue
                    existing_pos = np.array(map_points[point_id].position)
                    distance = np.linalg.norm(position_3d - existing_pos)
                    if distance < self.config.map.distance_threshold * 3:
                        return point_id
        return None

