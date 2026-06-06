#!/usr/bin/env python3
"""Device 0 stereo camera debug runner."""

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.stereo_slam import StereoSLAM  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run StereoSLAM against a side-by-side stereo camera.")
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera device id.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to process.")
    parser.add_argument("--width", type=int, default=2560, help="Side-by-side frame width.")
    parser.add_argument("--height", type=int, default=720, help="Frame height.")
    parser.add_argument("--baseline", type=float, default=0.065, help="Stereo baseline in meters.")
    parser.add_argument("--fov", type=float, default=100.0, help="Horizontal FOV per eye in degrees.")
    parser.add_argument(
        "--eye-order",
        choices=("left-right", "right-left"),
        default="right-left",
        help="Order of the two eye images inside the side-by-side camera frame.",
    )
    parser.add_argument(
        "--save-map",
        default=str(PROJECT_ROOT / ".runtime" / "device0_map.json"),
        help="Path to save the generated map JSON.",
    )
    return parser.parse_args()


def open_camera(device_id: int, width: int, height: int):
    cap = cv2.VideoCapture(device_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def normalize_camera_frame(frame, expected_width: int, expected_height: int):
    """Crop camera frames to the expected side-by-side input size."""
    height, width = frame.shape[:2]
    if width != expected_width:
        raise ValueError(
            f"Camera returned width {width}, expected {expected_width}. "
            "Check camera mode or pass --width."
        )
    if height == expected_height:
        return frame
    if height > expected_height:
        top = (height - expected_height) // 2
        return frame[top:top + expected_height, :].copy()
    raise ValueError(
        f"Camera returned height {height}, expected at least {expected_height}. "
        "Check camera mode or pass --height."
    )


def main():
    args = parse_args()
    eye_width = args.width // 2
    save_map_path = Path(args.save_map)

    slam = StereoSLAM(
        device_id=args.device,
        baseline=args.baseline,
        image_width=eye_width,
        image_height=args.height,
        stereo_width=args.width,
        fov_horizontal=args.fov,
        eye_order=args.eye_order,
        debug_mode=False,
    )

    cap = open_camera(args.device, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera device {args.device}")

    processed = 0
    started = time.time()

    try:
        while processed < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"frame={processed}: camera returned no frame")
                time.sleep(0.05)
                continue

            frame = normalize_camera_frame(frame, args.width, args.height)

            result = slam.process_stereo_image(frame)
            processed += 1
            print(
                "frame={frame_id} ok={success} matches={matches} tri={tri} "
                "new={new} updated={updated} map_points={points} vo_inliers={inliers}".format(
                    frame_id=result["frame_id"],
                    success=result["success"],
                    matches=result.get("num_matches", 0),
                    tri=result.get("num_triangulated_points", 0),
                    new=result.get("num_new_points", 0),
                    updated=result.get("num_updated_points", 0),
                    points=result.get("total_map_points", 0),
                    inliers=result.get("vo_inliers", 0),
                )
            )
    finally:
        cap.release()

    slam.save_map(save_map_path)
    elapsed = max(time.time() - started, 1e-6)
    stats = slam.get_map_statistics()
    print(f"processed={processed} fps={processed / elapsed:.2f}")
    print(f"saved_map={save_map_path}")
    print(f"map_points={stats['num_points']} keyframes={stats['num_keyframes']}")


if __name__ == "__main__":
    main()
