"""Flask debug app for live StereoSLAM visualization."""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from src.core.stereo_slam import StereoSLAM
from .camera import normalize_camera_frame, open_stereo_camera
from .synthetic import create_synthetic_stereo_frame
from .templates import INDEX_HTML

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_MAP_PATH = PROJECT_ROOT / ".runtime" / "web_live_map.json"


@dataclass
class WebSLAMState:
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    shared_frame: object = None
    latest_result: dict = None
    processed_frame_count: int = 0
    running: bool = True
    cap: object = None
    slam: StereoSLAM = None
    save_map_path: Optional[Path] = DEFAULT_WEB_MAP_PATH
    save_every_n_frames: int = 30
    last_saved_frame_count: int = 0
    last_save_path: Optional[str] = None
    last_save_error: Optional[str] = None
    last_save_time: Optional[float] = None


def create_app(state: WebSLAMState = None) -> Flask:
    state = state or WebSLAMState()
    app = Flask(__name__)
    app.config["slam_state"] = state

    @app.route("/")
    def index():
        return INDEX_HTML

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


def frame_generator(state: WebSLAMState):
    state.slam = StereoSLAM(
        device_id=0,
        baseline=0.065,
        image_width=1280,
        image_height=720,
        stereo_width=2560,
        fov_horizontal=100.0,
        eye_order="right-left",
        debug_mode=False,
    )

    state.cap = open_stereo_camera(0, 2560, 720)
    use_synthetic = not state.cap.isOpened()

    if not use_synthetic:
        ok, test_frame = state.cap.read()
        if ok and test_frame is not None:
            logger.info(f"Camera opened at {test_frame.shape[1]}x{test_frame.shape[0]}")
        else:
            logger.warning("Camera opened but returned no frame; using synthetic frames")
            use_synthetic = True
    else:
        logger.warning("Unable to open camera; using synthetic frames")

    frame_count = 0
    while state.running:
        stereo_frame = _read_stereo_frame(state, use_synthetic)
        left_img, right_img = state.slam.split_stereo_image(stereo_frame)
        result = state.slam.process_stereo_image(stereo_frame)
        frame_count += 1

        display_img = draw_observations(
            left_img,
            right_img,
            state.slam.get_last_observations(),
            len(state.slam.map.get_3d_points_array()),
            state.slam.get_camera_position(),
        )

        with state.frame_lock:
            state.shared_frame = display_img
            state.latest_result = result
            state.processed_frame_count = frame_count

        if should_save_map(state, frame_count):
            save_current_map(state, frame_count)

        time.sleep(0.033)


def _read_stereo_frame(state: WebSLAMState, use_synthetic: bool):
    if not use_synthetic:
        ok, frame = state.cap.read()
        if ok and frame is not None:
            normalized = normalize_camera_frame(frame)
            if normalized is not None:
                return normalized

    return create_synthetic_stereo_frame(eye_order=state.slam.eye_order)


def draw_observations(left_img, right_img, observations, num_map_points: int, camera_position):
    display_img = np.hstack((left_img.copy(), right_img.copy()))
    mid_w = left_img.shape[1]

    for observation in observations:
        left_x, left_y = int(observation["left"][0]), int(observation["left"][1])
        right_x, right_y = int(observation["right"][0]), int(observation["right"][1])
        depth = float(observation["depth"])
        left_color, right_color = _depth_colors(depth)

        if 0 <= left_x < left_img.shape[1] and 0 <= left_y < left_img.shape[0]:
            cv2.circle(display_img, (left_x, left_y), 3, left_color, -1)
            cv2.circle(display_img, (left_x, left_y), 6, left_color, 1)

        if 0 <= right_x < right_img.shape[1] and 0 <= right_y < right_img.shape[0]:
            cv2.circle(display_img, (right_x + mid_w, right_y), 3, right_color, -1)
            cv2.circle(display_img, (right_x + mid_w, right_y), 6, right_color, 1)

    cv2.putText(
        display_img,
        f"Map Points: {num_map_points}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display_img,
        f"Observed: {len(observations)}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        display_img,
        f"Cam: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cv2.putText(display_img, "LEFT", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(
        display_img,
        "RIGHT",
        (mid_w + 20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return display_img


def _depth_colors(depth: float):
    if depth < 3:
        return (0, 255, 0), (255, 0, 0)
    if depth < 6:
        return (0, 255, 255), (255, 255, 0)
    return (0, 120, 255), (120, 120, 255)


def get_frame(state: WebSLAMState):
    with state.frame_lock:
        if state.shared_frame is None:
            return None
        success, buffer = cv2.imencode(
            ".jpg",
            state.shared_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        return buffer.tobytes() if success else None


def get_stats(state: WebSLAMState):
    with state.frame_lock:
        if state.slam is None:
            return {
                "frame_count": 0,
                "num_points": 0,
                "num_current_observations": 0,
                "camera_pos": [0.0, 0.0, 0.0],
                "last_result": None,
            }

        positions = state.slam.map.get_3d_points_array()
        latest = _json_safe(state.latest_result or {})
        return {
            "frame_count": state.processed_frame_count,
            "num_points": len(positions),
            "num_current_observations": latest.get("num_current_observations", 0),
            "camera_pos": _json_safe(state.slam.get_camera_position()),
            "last_result": latest,
            "map_save_path": str(state.save_map_path) if state.save_map_path else None,
            "last_saved_frame_count": state.last_saved_frame_count,
            "last_save_error": state.last_save_error,
        }


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


def should_save_map(state: WebSLAMState, frame_count: int) -> bool:
    if state.save_map_path is None:
        return False
    if state.save_every_n_frames <= 0:
        return False
    return frame_count % state.save_every_n_frames == 0


def save_current_map(state: WebSLAMState, frame_count: Optional[int] = None):
    if state.save_map_path is None or state.slam is None:
        return

    try:
        state.slam.save_map(state.save_map_path)
        state.last_saved_frame_count = (
            frame_count if frame_count is not None else state.processed_frame_count
        )
        state.last_save_path = str(state.save_map_path)
        state.last_save_error = None
        state.last_save_time = time.time()
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        state.last_save_error = str(exc)
        logger.exception("Unable to save map to %s", state.save_map_path)


def cleanup(state: WebSLAMState):
    state.running = False
    save_current_map(state)
    if state.cap:
        state.cap.release()
    cv2.destroyAllWindows()


def run_web_slam(
    host: str = "0.0.0.0",
    port: int = 9704,
    save_map_path: Optional[Union[str, Path]] = DEFAULT_WEB_MAP_PATH,
    save_every_n_frames: int = 30,
):
    logging.basicConfig(level=logging.INFO)
    state = WebSLAMState(
        save_map_path=Path(save_map_path) if save_map_path else None,
        save_every_n_frames=save_every_n_frames,
    )
    app = create_app(state)
    thread = threading.Thread(target=frame_generator, args=(state,), daemon=True)
    thread.start()
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        cleanup(state)
