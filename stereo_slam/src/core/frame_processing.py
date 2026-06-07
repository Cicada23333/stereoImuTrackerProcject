"""Per-frame SLAM processing for StereoSLAM."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FrameProcessingMixin:
    """Feature extraction, stereo matching, VO, and map update pipeline."""

    def process_frame(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> Dict:
        """Process already split left/right images and update the map."""
        if left_image.shape[:2] != right_image.shape[:2]:
            raise ValueError(
                f"Left/right image sizes must match, left={left_image.shape[:2]}, "
                f"right={right_image.shape[:2]}"
            )
        if left_image.shape[1] != self.image_width or left_image.shape[0] != self.image_height:
            raise ValueError(
                f"Single-eye image size mismatch: expected {self.image_width}x{self.image_height}, "
                f"got {left_image.shape[1]}x{left_image.shape[0]}"
            )

        if frame_id is None:
            frame_id = self.map.frame_counter
        self.map.frame_counter = max(self.map.frame_counter, frame_id + 1)
        self.last_observations = []

        left_keypoints, right_keypoints, left_descriptors, right_descriptors = (
            self.feature_extractor.extract_stereo(left_image, right_image)
        )
        self.logger.info(
            f"  Extracted {len(left_keypoints)} left and {len(right_keypoints)} right keypoints"
        )

        if not left_keypoints or not right_keypoints:
            return {
                "frame_id": frame_id,
                "success": False,
                "error": "No features detected",
                "num_matches": 0,
            }

        matches = self.stereo_matcher.match_stereo_rectified(
            left_keypoints,
            right_keypoints,
            left_descriptors,
            right_descriptors,
            max_vertical_diff=self.config.map.max_vertical_disparity,
            min_disparity=self.config.map.min_disparity,
            max_disparity=self.config.map.max_disparity,
        )
        self.logger.info(f"  Found {len(matches)} stereo matches")

        if not matches:
            return {
                "frame_id": frame_id,
                "success": False,
                "error": "No stereo matches found",
                "num_matches": 0,
            }

        triangulated_points = self._triangulate_with_quality_check(
            left_keypoints, right_keypoints, matches
        )
        self.logger.info(f"  Triangulated {len(triangulated_points)} valid 3D points")

        num_matches_with_map, num_inliers = 0, 0
        camera_moved_significant = False
        camera_movement_distance = 0.0

        if frame_id > 0 and self.current_3d_points is not None:
            pose, num_matches_with_map, num_inliers = self.visual_odometry.update(
                left_keypoints, left_descriptors
            )
            self.camera_pose = pose

            current_pos = self._camera_center_world()
            if hasattr(self, "_prev_camera_pos"):
                camera_movement_distance = np.linalg.norm(current_pos - self._prev_camera_pos)
                camera_moved_significant = (
                    camera_movement_distance > self.config.map.min_keyframe_distance
                )
            self._prev_camera_pos = current_pos.copy()
            self.logger.info(
                f"  VO: {num_inliers} inliers from {num_matches_with_map} matches, "
                f"camera moved {camera_movement_distance:.3f}m"
            )

        triangulated_world_points = self._to_world_triangulated_points(triangulated_points)
        new_points_count, updated_points_count = self._update_map_points(
            triangulated_world_points,
            left_keypoints,
            left_image,
            frame_id,
            camera_moved_significant,
        )

        if camera_moved_significant or frame_id == 0:
            self.map.add_keyframe(
                frame_id=frame_id,
                left_image=left_image,
                right_image=right_image,
                left_keypoints=left_keypoints,
                right_keypoints=right_keypoints,
                left_descriptors=left_descriptors,
                right_descriptors=right_descriptors,
                camera_pose=self.camera_pose.copy(),
            )
            self.logger.info(f"  Added keyframe at frame {frame_id}")
        else:
            self.logger.debug("  Skipped keyframe (camera not moved enough)")

        self._update_vo_cache(left_keypoints, left_descriptors, triangulated_points)

        if frame_id > 0 and frame_id % 50 == 0:
            culled_count = self.map.cull_insecure_points()
            if culled_count > 0:
                self.logger.info(f"  Culled {culled_count} insecure points")

        result = {
            "frame_id": frame_id,
            "success": True,
            "num_keypoints_left": len(left_keypoints),
            "num_keypoints_right": len(right_keypoints),
            "num_matches": len(matches),
            "num_triangulated_points": len(triangulated_points),
            "num_current_observations": len(self.last_observations),
            "num_new_points": new_points_count,
            "num_updated_points": updated_points_count,
            "total_map_points": len(self.map.points),
            "camera_pose": self.camera_pose.tolist(),
            "vo_matches": num_matches_with_map,
            "vo_inliers": num_inliers,
            "camera_moved_significant": camera_moved_significant,
            "camera_movement_distance": camera_movement_distance,
            "timestamp": datetime.now().isoformat(),
        }

        if self.debug_mode:
            result["debug_info"] = self._get_debug_info(left_image, left_keypoints, matches)

        return result

    def _update_vo_cache(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        triangulated_points: List[Tuple[int, np.ndarray]],
    ):
        if not triangulated_points:
            return

        positions = np.array([p[1] for p in triangulated_points], dtype=np.float32)
        valid_indices = [p[0] for p in triangulated_points]

        max_cache_size = self.config.map.max_cache_size
        if len(positions) > max_cache_size:
            indices = np.random.choice(len(positions), max_cache_size, replace=False)
            positions = positions[indices]
            valid_indices = [valid_indices[i] for i in indices]

        self.current_3d_points = positions
        self.current_left_keypoints = [keypoints[i] for i in valid_indices if i < len(keypoints)]
        self.current_left_descriptors = descriptors[valid_indices] if descriptors is not None else None

        if len(self.current_3d_points) > 10:
            if self.visual_odometry.prev_3d_points is None:
                self.visual_odometry.initialize(
                    self.current_left_keypoints,
                    self.current_left_descriptors,
                    self.current_3d_points,
                )
            else:
                self.visual_odometry.prev_keypoints = self.current_left_keypoints
                self.visual_odometry.prev_descriptors = self.current_left_descriptors
                self.visual_odometry.prev_3d_points = self.current_3d_points

    def _get_debug_info(
        self,
        left_image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> Dict:
        debug_img = left_image.copy()

        for keypoint in keypoints[:100]:
            cv2.circle(debug_img, (int(keypoint.pt[0]), int(keypoint.pt[1])), 3, (0, 255, 0), -1)

        for match in matches[:50]:
            point = (
                int(keypoints[match.queryIdx].pt[0]),
                int(keypoints[match.queryIdx].pt[1]),
            )
            cv2.circle(debug_img, point, 5, (255, 0, 0), -1)

        return {
            "debug_image_shape": debug_img.shape,
            "matched_points_displayed": min(50, len(matches)),
        }

