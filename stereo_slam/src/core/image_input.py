"""High-level stereo image input helpers for StereoSLAM."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import cv2
import numpy as np


class StereoImageInputMixin:
    """Input APIs for side-by-side stereo frames."""

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image

        image_path = Path(image)
        loaded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return loaded

    def split_stereo_image(self, stereo_image: Union[str, Path, np.ndarray]):
        """Split a side-by-side stereo frame into left and right eye images."""
        image = self._load_image(stereo_image)
        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be grayscale or BGR/BGRA, got ndim={image.ndim}")

        height, width = image.shape[:2]
        if width != self.stereo_width or height != self.image_height:
            raise ValueError(
                f"Stereo frame size mismatch: expected {self.stereo_width}x{self.image_height}, "
                f"got {width}x{height}. Initialize StereoSLAM with matching dimensions if needed."
            )

        if width % 2 != 0:
            raise ValueError(f"Stereo frame width must be even, got {width}")

        mid = width // 2
        first_half = image[:, :mid]
        second_half = image[:, mid:]
        if self.eye_order == "left-right":
            return first_half.copy(), second_half.copy()
        return second_half.copy(), first_half.copy()

    def process_stereo_image(
        self,
        stereo_image: Union[str, Path, np.ndarray],
        frame_id: Optional[int] = None,
        save_map_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Process one side-by-side stereo frame and update the map."""
        input_image = self._load_image(stereo_image)
        left_image, right_image = self.split_stereo_image(input_image)
        result = self.process_frame(left_image, right_image, frame_id=frame_id)
        result["input_shape"] = list(input_image.shape)
        result["left_shape"] = list(left_image.shape)
        result["right_shape"] = list(right_image.shape)
        result["eye_order"] = self.eye_order
        self._save_map_if_requested(save_map_path)
        return result

    def process_images(
        self,
        images: Iterable[Union[str, Path, np.ndarray]],
        save_map_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict]:
        """Process multiple side-by-side stereo frames in order."""
        results = []
        for image in images:
            results.append(self.process_stereo_image(image, save_map_path=None))

        self._save_map_if_requested(save_map_path)
        return results

    def process(
        self,
        images: Union[str, Path, np.ndarray, Iterable[Union[str, Path, np.ndarray]]],
        save_map_path: Optional[Union[str, Path]] = None,
    ):
        """Unified input API: one image returns a dict, many images return list[dict]."""
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return self.process_images(list(images), save_map_path=save_map_path)
        if isinstance(images, (str, Path, np.ndarray)):
            return self.process_stereo_image(images, save_map_path=save_map_path)
        return self.process_images(images, save_map_path=save_map_path)

    def _save_map_if_requested(self, save_map_path: Optional[Union[str, Path]] = None):
        target = Path(save_map_path) if save_map_path else self.auto_save_path
        if target is not None:
            self.save_map(target)

