#!/usr/bin/env python3
"""Compatibility entrypoint for the StereoSLAM debug web server."""

import argparse
from pathlib import Path

from src.web import (
    DEFAULT_WEB_MAP_PATH,
    create_synthetic_stereo_frame,
    create_synthetic_stereo_images,
    run_web_slam,
)

__all__ = [
    "DEFAULT_WEB_MAP_PATH",
    "create_synthetic_stereo_frame",
    "create_synthetic_stereo_images",
    "run_web_slam",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the StereoSLAM debug web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask bind host.")
    parser.add_argument("--port", type=int, default=9704, help="Flask bind port.")
    parser.add_argument(
        "--save-map",
        default=str(DEFAULT_WEB_MAP_PATH),
        help="Path for the auto-saved map JSON.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="Save the map every N processed frames. Use 0 to save only on shutdown.",
    )
    parser.add_argument(
        "--no-save-map",
        action="store_true",
        help="Disable periodic and shutdown map saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_map_path = None if args.no_save_map else Path(args.save_map)

    print("=" * 60)
    print("Stereo SLAM Web Visualization")
    print("=" * 60)
    print(f"Open http://localhost:{args.port}")
    if save_map_path:
        print(f"Auto-save map: {save_map_path} every {args.save_every} frames")
    else:
        print("Auto-save map: disabled")
    print("Press Ctrl+C to exit")
    print("=" * 60)
    run_web_slam(
        host=args.host,
        port=args.port,
        save_map_path=save_map_path,
        save_every_n_frames=args.save_every,
    )
