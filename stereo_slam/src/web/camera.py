"""Camera helpers used by the web and CLI debug runners."""

import cv2


def open_stereo_camera(device_id: int = 0, width: int = 2560, height: int = 720):
    """Open a side-by-side stereo camera and request the desired mode."""
    cap = cv2.VideoCapture(device_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def normalize_camera_frame(frame, expected_width: int = 2560, expected_height: int = 720):
    """Crop camera output to the library's expected side-by-side frame size."""
    height, width = frame.shape[:2]
    if width != expected_width:
        return None
    if height == expected_height:
        return frame
    if height > expected_height:
        top = (height - expected_height) // 2
        return frame[top:top + expected_height, :].copy()
    return None

