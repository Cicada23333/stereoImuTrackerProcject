"""Read-only localization against a saved StereoSLAM map."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ..features import FeatureExtractor, StereoMatcher
from ..geometry import GeometryUtils
from ..map import Map


class StereoMapLocalizer:
    """Localize a stereo frame against a saved map without updating that map."""

    def __init__(
        self,
        map_path: Optional[Union[str, Path]] = None,
        device_id: int = 0,
        baseline: float = 0.065,
        focal_length: Optional[float] = None,
        image_width: int = 1280,
        image_height: int = 720,
        stereo_width: Optional[int] = None,
        fov_horizontal: float = 100.0,
        eye_order: str = "left-right",
        ratio_threshold: float = 0.75,
        min_pnp_matches: int = 6,
        min_pnp_inliers: int = 8,
        strong_pose_min_inliers: int = 25,
        min_inliers_ratio: float = 0.04,
        ransac_reproj_threshold: float = 4.0,
        pnp_iterations: int = 1000,
        max_descriptor_distance: float = 72.0,
        max_pnp_matches: int = 600,
        require_reciprocal_match: bool = True,
        use_stereo_filter: bool = False,
    ):
        self.device_id = device_id
        self.baseline = baseline
        self.image_width = image_width
        self.image_height = image_height
        self.stereo_width = stereo_width or image_width * 2
        if eye_order not in ("left-right", "right-left"):
            raise ValueError('eye_order must be "left-right" or "right-left"')
        self.eye_order = eye_order
        self.ratio_threshold = ratio_threshold
        self.min_pnp_matches = min_pnp_matches
        self.min_pnp_inliers = min_pnp_inliers
        self.strong_pose_min_inliers = strong_pose_min_inliers
        self.min_inliers_ratio = min_inliers_ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.pnp_iterations = pnp_iterations
        self.max_descriptor_distance = max_descriptor_distance
        self.max_pnp_matches = max_pnp_matches
        self.require_reciprocal_match = require_reciprocal_match
        self.use_stereo_filter = use_stereo_filter

        focal_from_fov = image_width / (2 * np.tan(np.deg2rad(fov_horizontal / 2)))
        self.focal_length = float(focal_length) if focal_length is not None else float(focal_from_fov)
        self.principal_point = (image_width / 2, image_height / 2)
        self.K = np.array(
            [
                [self.focal_length, 0, self.principal_point[0]],
                [0, self.focal_length, self.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.distortion_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.feature_extractor = FeatureExtractor()
        self.stereo_matcher = StereoMatcher(ratio_threshold=ratio_threshold)
        self.map = Map(device_id=device_id)
        self.map_path: Optional[Path] = None
        self.camera_pose = np.eye(4, dtype=np.float64)

        self._map_point_ids: List[int] = []
        self._map_positions = np.empty((0, 3), dtype=np.float32)
        self._map_descriptors = np.empty((0, 32), dtype=np.uint8)

        if map_path is not None:
            self.load_map(map_path)

    def load_map(self, filepath: Union[str, Path]) -> Dict:
        self.map_path = Path(filepath)
        self.map = Map.load_from_file(str(self.map_path))
        self._refresh_map_index()
        return self.get_map_statistics()

    def get_map_statistics(self) -> Dict:
        stats = self.map.get_statistics()
        stats["map_path"] = str(self.map_path) if self.map_path else None
        stats["num_described_points"] = len(self._map_point_ids)
        return stats

    def split_stereo_image(self, stereo_image: Union[str, Path, np.ndarray]):
        image = self._load_image(stereo_image)
        height, width = image.shape[:2]
        if width != self.stereo_width or height != self.image_height:
            raise ValueError(
                f"Stereo frame size mismatch: expected {self.stereo_width}x{self.image_height}, "
                f"got {width}x{height}."
            )

        mid = width // 2
        first_half = image[:, :mid]
        second_half = image[:, mid:]
        if self.eye_order == "left-right":
            return first_half.copy(), second_half.copy()
        return second_half.copy(), first_half.copy()

    def localize_stereo_image(
        self,
        stereo_image: Union[str, Path, np.ndarray],
        frame_id: Optional[int] = None,
    ) -> Dict:
        input_image = self._load_image(stereo_image)
        left_image, right_image = self.split_stereo_image(input_image)
        result = self.localize_frame(left_image, right_image, frame_id=frame_id)
        result["input_shape"] = list(input_image.shape)
        result["left_shape"] = list(left_image.shape)
        result["right_shape"] = list(right_image.shape)
        result["eye_order"] = self.eye_order
        return result

    def localize_frame(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> Dict:
        if frame_id is None:
            frame_id = 0

        left_keypoints, right_keypoints, left_descriptors, right_descriptors = (
            self.feature_extractor.extract_stereo(left_image, right_image)
        )
        stereo_matches = self.stereo_matcher.match_stereo_rectified(
            left_keypoints,
            right_keypoints,
            left_descriptors,
            right_descriptors,
            max_vertical_diff=20.0,
            min_disparity=2.0,
            max_disparity=300.0,
        )

        base = {
            "frame_id": frame_id,
            "success": False,
            "num_map_points": len(self.map.points),
            "num_described_map_points": len(self._map_point_ids),
            "num_keypoints_left": len(left_keypoints),
            "num_keypoints_right": len(right_keypoints),
            "num_stereo_matches": len(stereo_matches),
            "num_candidate_features": 0,
            "num_raw_descriptor_matches": 0,
            "num_ratio_matches": 0,
            "num_distance_matches": 0,
            "num_reciprocal_matches": 0,
            "num_map_matches": 0,
            "num_pnp_used_matches": 0,
            "num_pnp_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_inlier_reprojection_error": None,
            "median_inlier_reprojection_error": None,
            "descriptor_distance_median": None,
            "camera_pose": self.camera_pose.tolist(),
            "camera_position": self.get_camera_position().tolist(),
            "candidate_camera_pose": None,
            "candidate_camera_position": None,
            "quality": "not_localized",
            "matched_map_points": [],
            "visible_map_points": [],
            "timestamp": datetime.now().isoformat(),
        }

        if len(self._map_point_ids) < self.min_pnp_matches:
            base["error"] = (
                "Loaded map has too few ORB descriptors for localization. "
                "Regenerate the map with the current code so points contain descriptor fields."
            )
            return base
        if left_descriptors is None or len(left_keypoints) < self.min_pnp_matches:
            base["error"] = "Not enough current-frame ORB features"
            return base

        candidate_indices = self._candidate_left_indices(stereo_matches, len(left_keypoints))
        base["num_candidate_features"] = len(candidate_indices)
        if len(candidate_indices) < self.min_pnp_matches:
            base["error"] = "Not enough candidate features for map matching"
            return base

        matches, match_debug = self._match_map_descriptors(left_descriptors, candidate_indices)
        base.update(match_debug)
        base["num_map_matches"] = len(matches)
        if len(matches) < self.min_pnp_matches:
            base["error"] = f"Not enough map descriptor matches: {len(matches)}"
            return base

        object_points = np.asarray([item["position"] for item in matches], dtype=np.float32)
        image_points = np.asarray([left_keypoints[item["keypoint_index"]].pt for item in matches], dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.K,
            self.distortion_coeffs,
            iterationsCount=self.pnp_iterations,
            reprojectionError=self.ransac_reproj_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None:
            base["error"] = "PnP failed"
            return base

        inlier_indices = inliers.reshape(-1)
        num_inliers = len(inlier_indices)
        inlier_ratio = num_inliers / max(len(matches), 1)
        base["num_pnp_inliers"] = num_inliers
        base["inlier_ratio"] = inlier_ratio

        if num_inliers >= 6 and hasattr(cv2, "solvePnPRefineLM"):
            inlier_object_points = object_points[inlier_indices]
            inlier_image_points = image_points[inlier_indices]
            rvec, tvec = cv2.solvePnPRefineLM(
                inlier_object_points,
                inlier_image_points,
                self.K,
                self.distortion_coeffs,
                rvec,
                tvec,
            )

        candidate_pose = self._pose_from_rvec_tvec(rvec, tvec)
        candidate_position = self._camera_position_from_pose(candidate_pose)
        reprojection_errors = self._reprojection_errors(object_points, image_points, rvec, tvec)
        inlier_errors = reprojection_errors[inlier_indices]
        base["mean_inlier_reprojection_error"] = float(np.mean(inlier_errors))
        base["median_inlier_reprojection_error"] = float(np.median(inlier_errors))
        base["candidate_camera_pose"] = candidate_pose.tolist()
        base["candidate_camera_position"] = candidate_position.tolist()

        inlier_set = set(int(index) for index in inlier_indices)
        matched_points = self._format_matched_points(
            matches,
            left_keypoints,
            object_points,
            image_points,
            rvec,
            tvec,
            inlier_set,
            candidate_pose,
        )
        visible_points = self.project_map_points(candidate_pose, left_image.shape, max_points=1200)
        base["matched_map_points"] = matched_points
        base["visible_map_points"] = visible_points

        enough_inliers = num_inliers >= self.min_pnp_inliers
        enough_ratio = inlier_ratio >= self.min_inliers_ratio
        strong_pose = num_inliers >= self.strong_pose_min_inliers
        if not enough_inliers or not (enough_ratio or strong_pose):
            base["error"] = (
                f"PnP rejected: inliers={num_inliers}/{len(matches)} "
                f"ratio={inlier_ratio:.1%}, need {self.min_pnp_inliers}+ inliers and "
                f"{self.min_inliers_ratio:.1%} ratio or {self.strong_pose_min_inliers}+ strong inliers"
            )
            return base

        self.camera_pose = candidate_pose.copy()
        quality = "strong" if strong_pose else "ratio"

        base.update(
            {
                "success": True,
                "error": None,
                "camera_pose": self.camera_pose.tolist(),
                "camera_position": self.get_camera_position().tolist(),
                "quality": quality,
            }
        )
        return base

    def get_camera_position(self) -> np.ndarray:
        return self._camera_position_from_pose(self.camera_pose)

    def project_map_points(
        self,
        camera_pose: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (720, 1280),
        max_points: int = 1200,
    ) -> List[Dict]:
        if camera_pose is None:
            camera_pose = self.camera_pose
        if not self.map.points:
            return []

        point_ids = list(self.map.points.keys())
        positions = np.asarray([point.position for point in self.map.points.values()], dtype=np.float32)
        projected, indices = GeometryUtils.project_3d_to_2d_with_depth(
            positions,
            camera_pose,
            self.K,
            image_shape,
            return_indices=True,
        )

        items = []
        for projected_point, point_index in zip(projected, indices):
            point_id = point_ids[int(point_index)]
            if len(items) >= max_points:
                break
            point = self.map.points[point_id]
            right_projection = self._project_right_point(point.position, camera_pose)
            items.append(
                {
                    "point_id": int(point_id),
                    "left": [float(projected_point[0]), float(projected_point[1])],
                    "right": right_projection,
                    "depth": float(projected_point[2]),
                    "color": point.color.tolist() if point.color is not None else None,
                    "has_descriptor": point.descriptor is not None,
                }
            )
        return items

    def _refresh_map_index(self):
        self._map_point_ids, self._map_positions, self._map_descriptors = self.map.get_described_points()

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        image_path = Path(image)
        loaded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return loaded

    def _candidate_left_indices(self, stereo_matches: Sequence[cv2.DMatch], num_keypoints: int) -> List[int]:
        if self.use_stereo_filter:
            return sorted({match.queryIdx for match in stereo_matches if match.queryIdx < num_keypoints})
        return list(range(num_keypoints))

    def _match_map_descriptors(
        self,
        current_descriptors: np.ndarray,
        candidate_indices: Sequence[int],
    ) -> Tuple[List[Dict], Dict]:
        candidate_indices = [index for index in candidate_indices if index < len(current_descriptors)]
        debug = {
            "num_raw_descriptor_matches": 0,
            "num_ratio_matches": 0,
            "num_distance_matches": 0,
            "num_reciprocal_matches": 0,
            "num_pnp_used_matches": 0,
            "descriptor_distance_median": None,
        }
        if len(candidate_indices) < 2 or len(self._map_descriptors) < 2:
            return [], debug

        candidate_descriptors = np.asarray(current_descriptors[candidate_indices], dtype=np.uint8)
        map_descriptors = np.asarray(self._map_descriptors, dtype=np.uint8)
        if map_descriptors.shape[1] != candidate_descriptors.shape[1]:
            return [], debug

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(candidate_descriptors, map_descriptors, k=2)
        debug["num_raw_descriptor_matches"] = sum(1 for pair in raw_matches if len(pair) >= 2)

        reverse_best = {}
        if self.require_reciprocal_match:
            for match in matcher.match(map_descriptors, candidate_descriptors):
                reverse_best[match.queryIdx] = candidate_indices[match.trainIdx]

        best_by_map = {}
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance >= self.ratio_threshold * second.distance:
                continue
            debug["num_ratio_matches"] += 1
            if first.distance > self.max_descriptor_distance:
                continue
            debug["num_distance_matches"] += 1

            current_index = candidate_indices[first.queryIdx]
            map_descriptor_index = first.trainIdx
            if (
                self.require_reciprocal_match
                and reverse_best.get(map_descriptor_index) != current_index
            ):
                continue
            debug["num_reciprocal_matches"] += 1

            previous = best_by_map.get(map_descriptor_index)
            if previous is None or first.distance < previous["distance"]:
                previous = {
                    "map_descriptor_index": map_descriptor_index,
                    "keypoint_index": current_index,
                    "distance": float(first.distance),
                    "second_best_distance": float(second.distance),
                    "ratio": float(first.distance / max(second.distance, 1e-6)),
                    "point_id": self._map_point_ids[map_descriptor_index],
                    "position": self._map_positions[map_descriptor_index],
                }
                best_by_map[map_descriptor_index] = previous

        matches = list(best_by_map.values())
        matches.sort(key=lambda item: item["distance"])
        if self.max_pnp_matches > 0:
            matches = matches[: self.max_pnp_matches]
        debug["num_pnp_used_matches"] = len(matches)
        if matches:
            debug["descriptor_distance_median"] = float(
                np.median([item["distance"] for item in matches])
            )
        return matches, debug

    def _format_matched_points(
        self,
        matches: Sequence[Dict],
        keypoints: Sequence[cv2.KeyPoint],
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        inlier_indices: set,
        camera_pose: np.ndarray,
    ) -> List[Dict]:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.K,
            self.distortion_coeffs,
        )
        projected = projected.reshape(-1, 2)
        formatted = []
        for index, item in enumerate(matches):
            left_pt = image_points[index]
            projected_left = projected[index]
            position = np.asarray(item["position"], dtype=np.float32)
            formatted.append(
                {
                    "point_id": int(item["point_id"]),
                    "map_position": position.tolist(),
                    "left_keypoint": [float(left_pt[0]), float(left_pt[1])],
                    "projected_left": [float(projected_left[0]), float(projected_left[1])],
                    "projected_right": self._project_right_point(position, camera_pose),
                    "reprojection_error": float(np.linalg.norm(left_pt - projected_left)),
                    "descriptor_distance": float(item["distance"]),
                    "descriptor_ratio": float(item["ratio"]),
                    "inlier": index in inlier_indices,
                    "keypoint_size": float(keypoints[item["keypoint_index"]].size),
                }
            )
        return formatted

    def _project_right_point(self, world_point: np.ndarray, camera_pose: np.ndarray) -> Optional[List[float]]:
        right_pose = camera_pose.copy()
        right_pose[:3, 3] -= np.array([self.baseline, 0.0, 0.0])
        projected = GeometryUtils.project_3d_to_2d(
            np.asarray([world_point], dtype=np.float32),
            right_pose,
            self.K,
            (self.image_height, self.image_width),
        )
        if len(projected) == 0:
            return None
        return [float(projected[0, 0]), float(projected[0, 1])]

    def _pose_from_rvec_tvec(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = tvec.reshape(3)
        return pose

    def _camera_position_from_pose(self, pose: np.ndarray) -> np.ndarray:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        return -rotation.T @ translation

    def _reprojection_errors(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.K,
            self.distortion_coeffs,
        )
        projected = projected.reshape(-1, 2)
        return np.linalg.norm(image_points - projected, axis=1)
