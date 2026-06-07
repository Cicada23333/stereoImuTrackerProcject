"""Read-only web test for localizing against an existing map."""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from src.core.localization import StereoMapLocalizer
from .app import DEFAULT_WEB_MAP_PATH
from .camera import normalize_camera_frame, open_stereo_camera
from .localization_templates import LOCALIZATION_INDEX_HTML

logger = logging.getLogger(__name__)
DEFAULT_LOCALIZATION_MAP_PATH = DEFAULT_WEB_MAP_PATH


@dataclass
class LocalizationWebState:
    map_path: Path = DEFAULT_LOCALIZATION_MAP_PATH
    device_id: int = 0
    baseline: float = 0.065
    image_width: int = 1280
    image_height: int = 720
    stereo_width: int = 2560
    fov_horizontal: float = 100.0
    eye_order: str = "right-left"
    use_stereo_filter: bool = False
    min_pnp_inliers: int = 8
    strong_pose_min_inliers: int = 25
    min_inliers_ratio: float = 0.04
    max_descriptor_distance: float = 72.0
    max_pnp_matches: int = 600
    require_reciprocal_match: bool = True
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    shared_frame: object = None
    latest_result: dict = None
    processed_frame_count: int = 0
    running: bool = True
    cap: object = None
    localizer: StereoMapLocalizer = None


def create_localization_app(state: LocalizationWebState = None) -> Flask:
    state = state or LocalizationWebState()
    app = Flask(__name__)
    app.config["localization_state"] = state

    @app.route("/")
    def index():
        return LOCALIZATION_INDEX_HTML

    @app.route("/frame.jpg")
    def frame():
        frame_data = get_frame(state)
        if frame_data:
            return Response(frame_data, mimetype="image/jpeg")
        return Response(b"", mimetype="image/jpeg")

    @app.route("/stats")
    def stats():
        return jsonify(get_stats(state))

    return app


def localization_frame_generator(state: LocalizationWebState):
    state.localizer = StereoMapLocalizer(
        map_path=state.map_path,
        device_id=state.device_id,
        baseline=state.baseline,
        image_width=state.image_width,
        image_height=state.image_height,
        stereo_width=state.stereo_width,
        fov_horizontal=state.fov_horizontal,
        eye_order=state.eye_order,
        use_stereo_filter=state.use_stereo_filter,
        min_pnp_inliers=state.min_pnp_inliers,
        strong_pose_min_inliers=state.strong_pose_min_inliers,
        min_inliers_ratio=state.min_inliers_ratio,
        max_descriptor_distance=state.max_descriptor_distance,
        max_pnp_matches=state.max_pnp_matches,
        require_reciprocal_match=state.require_reciprocal_match,
    )

    state.cap = open_stereo_camera(state.device_id, state.stereo_width, state.image_height)
    if not state.cap.isOpened():
        message = f"Unable to open camera device {state.device_id}"
        logger.error(message)
        _publish_error_frame(state, message)
        return

    frame_count = 0
    while state.running:
        ok, frame = state.cap.read()
        if not ok or frame is None:
            _publish_error_frame(state, "Camera returned no frame")
            time.sleep(0.05)
            continue

        normalized = normalize_camera_frame(frame, state.stereo_width, state.image_height)
        if normalized is None:
            raw_height, raw_width = frame.shape[:2]
            _publish_error_frame(
                state,
                f"Camera frame {raw_width}x{raw_height}, expected {state.stereo_width}x{state.image_height}",
            )
            time.sleep(0.05)
            continue

        left_img, right_img = state.localizer.split_stereo_image(normalized)
        result = state.localizer.localize_stereo_image(normalized, frame_id=frame_count)
        display_img = draw_localization(
            left_img,
            right_img,
            result,
            state.localizer.baseline,
        )
        frame_count += 1

        with state.frame_lock:
            state.shared_frame = display_img
            state.latest_result = result
            state.processed_frame_count = frame_count

        time.sleep(0.033)


def draw_localization(left_img, right_img, result: dict, baseline: float):
    display_img = np.hstack((left_img.copy(), right_img.copy()))
    mid_w = left_img.shape[1]

    for point in result.get("visible_map_points", []):
        left = point.get("left")
        if left:
            x, y = int(left[0]), int(left[1])
            if 0 <= x < left_img.shape[1] and 0 <= y < left_img.shape[0]:
                cv2.circle(display_img, (x, y), 1, (90, 90, 90), -1)
        right = point.get("right")
        if right:
            x, y = int(right[0] + mid_w), int(right[1])
            if mid_w <= x < display_img.shape[1] and 0 <= y < display_img.shape[0]:
                cv2.circle(display_img, (x, y), 1, (80, 80, 120), -1)

    inlier_count = 0
    for match in result.get("matched_map_points", []):
        if not match.get("inlier"):
            continue
        inlier_count += 1
        left_keypoint = match["left_keypoint"]
        projected_left = match["projected_left"]
        projected_right = match.get("projected_right")
        left_feature = (int(left_keypoint[0]), int(left_keypoint[1]))
        left_projection = (int(projected_left[0]), int(projected_left[1]))

        if _inside(left_projection, left_img.shape):
            cv2.circle(display_img, left_projection, 5, (0, 255, 0), 1)
        if _inside(left_feature, left_img.shape):
            cv2.circle(display_img, left_feature, 3, (0, 255, 255), -1)
        if _inside(left_projection, left_img.shape) and _inside(left_feature, left_img.shape):
            cv2.line(display_img, left_projection, left_feature, (0, 200, 255), 1)

        if projected_right:
            right_projection = (int(projected_right[0] + mid_w), int(projected_right[1]))
            if mid_w <= right_projection[0] < display_img.shape[1] and 0 <= right_projection[1] < display_img.shape[0]:
                cv2.circle(display_img, right_projection, 5, (255, 120, 0), 1)

    status = "LOCALIZED" if result.get("success") else "NO POSE"
    status_color = (0, 255, 0) if result.get("success") else (0, 80, 255)
    cam = result.get("camera_position", [0.0, 0.0, 0.0])
    inlier_ratio = result.get("inlier_ratio", 0.0) or 0.0
    median_error = result.get("median_inlier_reprojection_error")
    median_error_text = "--" if median_error is None else f"{median_error:.2f}px"
    lines = [
        (
            f"{status} {result.get('quality', '')} "
            f"inliers={result.get('num_pnp_inliers', 0)} "
            f"used={result.get('num_pnp_used_matches', 0)} "
            f"ratio={inlier_ratio:.1%}"
        ),
        f"Map {result.get('num_map_points', 0)} pts / {result.get('num_described_map_points', 0)} described",
        f"Cam ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f})",
        f"Matched projected points: {inlier_count}  median reproj={median_error_text}",
    ]
    if result.get("error"):
        lines.append(str(result["error"])[:120])

    y = 30
    for index, line in enumerate(lines):
        color = status_color if index == 0 else (0, 255, 255)
        cv2.putText(display_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 24

    cv2.putText(display_img, "LEFT", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_img, "RIGHT", (mid_w + 20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return display_img


def get_frame(state: LocalizationWebState):
    with state.frame_lock:
        if state.shared_frame is None:
            return None
        success, buffer = cv2.imencode(
            ".jpg",
            state.shared_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        return buffer.tobytes() if success else None


def get_stats(state: LocalizationWebState):
    with state.frame_lock:
        result = _compact_result_for_stats(state.latest_result or {})
        return {
            "frame_count": state.processed_frame_count,
            "map_path": str(state.map_path),
            "last_result": _json_safe(result),
        }


def cleanup_localization(state: LocalizationWebState):
    state.running = False
    if state.cap:
        state.cap.release()
    cv2.destroyAllWindows()


def run_localization_web(
    map_path: Union[str, Path] = DEFAULT_LOCALIZATION_MAP_PATH,
    host: str = "0.0.0.0",
    port: int = 9705,
    device_id: int = 0,
    baseline: float = 0.065,
    width: int = 2560,
    height: int = 720,
    fov_horizontal: float = 100.0,
    eye_order: str = "right-left",
    use_stereo_filter: bool = False,
    min_pnp_inliers: int = 8,
    strong_pose_min_inliers: int = 25,
    min_inliers_ratio: float = 0.04,
    max_descriptor_distance: float = 72.0,
    max_pnp_matches: int = 600,
    require_reciprocal_match: bool = True,
):
    logging.basicConfig(level=logging.INFO)
    state = LocalizationWebState(
        map_path=Path(map_path),
        device_id=device_id,
        baseline=baseline,
        image_width=width // 2,
        image_height=height,
        stereo_width=width,
        fov_horizontal=fov_horizontal,
        eye_order=eye_order,
        use_stereo_filter=use_stereo_filter,
        min_pnp_inliers=min_pnp_inliers,
        strong_pose_min_inliers=strong_pose_min_inliers,
        min_inliers_ratio=min_inliers_ratio,
        max_descriptor_distance=max_descriptor_distance,
        max_pnp_matches=max_pnp_matches,
        require_reciprocal_match=require_reciprocal_match,
    )
    app = create_localization_app(state)
    thread = threading.Thread(target=localization_frame_generator, args=(state,), daemon=True)
    thread.start()
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        cleanup_localization(state)


def _inside(point, image_shape):
    x, y = point
    return 0 <= x < image_shape[1] and 0 <= y < image_shape[0]


def _publish_error_frame(state: LocalizationWebState, message: str):
    frame = np.zeros((state.image_height, state.stereo_width, 3), dtype=np.uint8)
    cv2.putText(frame, message[:150], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    with state.frame_lock:
        state.shared_frame = frame
        state.latest_result = {
            "success": False,
            "error": message,
            "num_map_points": 0,
            "num_described_map_points": 0,
            "num_map_matches": 0,
            "num_pnp_inliers": 0,
            "camera_position": [0.0, 0.0, 0.0],
        }


def _compact_result_for_stats(result: dict):
    compact = dict(result)
    matched_points = compact.pop("matched_map_points", []) or []
    visible_points = compact.pop("visible_map_points", []) or []
    compact["num_matched_projected_points"] = sum(
        1 for point in matched_points if point.get("inlier")
    )
    compact["num_visible_projected_points"] = len(visible_points)
    return compact


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
