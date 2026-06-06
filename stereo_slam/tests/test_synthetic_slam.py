from simple_web_slam import create_synthetic_stereo_frame, create_synthetic_stereo_images
from src.core.stereo_slam import StereoSLAM
import numpy as np


def test_synthetic_frames_advance_counter_and_keep_initial_points():
    slam = StereoSLAM(debug_mode=False)
    left_image, right_image = create_synthetic_stereo_images()

    first = slam.process_frame(left_image, right_image)
    second = slam.process_frame(left_image, right_image)
    third = slam.process_frame(left_image, right_image)

    assert [first["frame_id"], second["frame_id"], third["frame_id"]] == [0, 1, 2]
    assert slam.map.frame_counter == 3
    assert first["success"] is True
    assert first["total_map_points"] > 0
    assert first["num_current_observations"] > 0
    assert len(slam.get_last_observations()) > 0
    assert third["total_map_points"] > 0


def test_side_by_side_frame_and_batch_update_map(tmp_path):
    slam = StereoSLAM(debug_mode=False)
    frame = create_synthetic_stereo_frame()
    map_path = tmp_path / "map.json"

    first = slam.process_stereo_image(frame)
    batch = slam.process_images([frame, frame], save_map_path=map_path)
    stacked_batch = slam.process(np.stack([frame, frame]))

    assert first["success"] is True
    assert first["input_shape"] == [720, 2560, 3]
    assert first["left_shape"] == [720, 1280, 3]
    assert first["right_shape"] == [720, 1280, 3]
    assert [result["frame_id"] for result in batch] == [1, 2]
    assert [result["frame_id"] for result in stacked_batch] == [3, 4]
    assert slam.get_map_statistics()["num_points"] > 0
    assert map_path.exists()

    loaded = StereoSLAM(debug_mode=False)
    loaded.load_map(map_path)
    assert loaded.map.frame_counter >= 3
