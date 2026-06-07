#!/usr/bin/env python3
"""Run the read-only map localization web test."""

import argparse
from pathlib import Path

from src.web import DEFAULT_LOCALIZATION_MAP_PATH, run_localization_web


def parse_args():
    parser = argparse.ArgumentParser(description="Run the read-only StereoSLAM map localization web test.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask bind host.")
    parser.add_argument("--port", type=int, default=9705, help="Flask bind port.")
    parser.add_argument("--device", type=int, default=0, help="Stereo camera device id.")
    parser.add_argument("--width", type=int, default=2560, help="Side-by-side camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Side-by-side camera frame height.")
    parser.add_argument("--baseline", type=float, default=0.065, help="Stereo baseline in meters.")
    parser.add_argument("--fov", type=float, default=100.0, help="Single-eye horizontal FOV in degrees.")
    parser.add_argument(
        "--eye-order",
        choices=("left-right", "right-left"),
        default="right-left",
        help="Order of the two eye images inside the side-by-side camera frame.",
    )
    parser.add_argument(
        "--map-path",
        default=str(DEFAULT_LOCALIZATION_MAP_PATH),
        help="Saved map JSON to load for read-only localization.",
    )
    parser.add_argument(
        "--stereo-filter",
        action="store_true",
        help="Only use current-frame features that also have a left/right stereo match.",
    )
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum PnP inliers required.")
    parser.add_argument(
        "--strong-inliers",
        type=int,
        default=25,
        help="Accept pose by absolute inlier count even when the global inlier ratio is low.",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.04,
        help="Minimum PnP inlier ratio unless --strong-inliers is reached.",
    )
    parser.add_argument(
        "--max-descriptor-distance",
        type=float,
        default=72.0,
        help="Maximum ORB Hamming distance for map descriptor matches.",
    )
    parser.add_argument(
        "--max-pnp-matches",
        type=int,
        default=600,
        help="Use only the best N descriptor matches for PnP.",
    )
    parser.add_argument(
        "--no-reciprocal-match",
        action="store_true",
        help="Disable reciprocal nearest-neighbor filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    map_path = Path(args.map_path)

    print("=" * 60)
    print("Stereo SLAM Read-only Map Localization")
    print("=" * 60)
    print(f"Map: {map_path}")
    print(f"Camera: device {args.device}, {args.width}x{args.height}, eye_order={args.eye_order}")
    print(f"Open http://localhost:{args.port}")
    print("Press Ctrl+C to exit")
    print("=" * 60)

    run_localization_web(
        map_path=map_path,
        host=args.host,
        port=args.port,
        device_id=args.device,
        baseline=args.baseline,
        width=args.width,
        height=args.height,
        fov_horizontal=args.fov,
        eye_order=args.eye_order,
        use_stereo_filter=args.stereo_filter,
        min_pnp_inliers=args.min_inliers,
        strong_pose_min_inliers=args.strong_inliers,
        min_inliers_ratio=args.min_inlier_ratio,
        max_descriptor_distance=args.max_descriptor_distance,
        max_pnp_matches=args.max_pnp_matches,
        require_reciprocal_match=not args.no_reciprocal_match,
    )
