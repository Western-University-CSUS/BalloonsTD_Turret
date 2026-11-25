from collections.abc import Sequence

import cv2
import numpy as np

from pinpoint.detectors.base import Box

HsvTriple = tuple[int, int, int]
HsvRange = tuple[HsvTriple, HsvTriple]

HSV_MAX = (179, 255, 255)

# Untuned starting points; replace with ranges from scripts/tune_hsv.py.
COLOR_PRESETS: dict[str, tuple[HsvRange, ...]] = {
    "cyan": (((85, 130, 130), (100, 255, 255)),),
    # Red wraps around hue 0/179, so it needs one range at each end.
    "red": (
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (179, 255, 255)),
    ),
}


class HsvDetector:
    """Detects blobs whose colour falls inside any of several HSV ranges."""

    def __init__(
        self,
        ranges: Sequence[HsvRange],
        min_area: float = 500.0,
        min_aspect: float = 0.5,
        max_aspect: float = 2.0,
    ):
        if not ranges:
            raise ValueError("at least one HSV range is required")
        for lower, upper in ranges:
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError(f"HSV bounds must have 3 channels, got {lower}, {upper}")
            for lo, hi, top in zip(lower, upper, HSV_MAX, strict=True):
                if not 0 <= lo <= hi <= top:
                    raise ValueError(f"invalid HSV range {lower}..{upper}, limits are {HSV_MAX}")
        if not 0 < min_aspect <= max_aspect:
            raise ValueError(f"need 0 < min_aspect <= max_aspect, got {min_aspect}, {max_aspect}")
        self.ranges = [
            (np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            for lower, upper in ranges
        ]
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "HsvDetector":
        """Build a detector from a named entry in COLOR_PRESETS."""
        if name not in COLOR_PRESETS:
            raise ValueError(f"unknown preset {name!r}, choose from {sorted(COLOR_PRESETS)}")
        return cls(COLOR_PRESETS[name], **kwargs)

    def detect_with_mask(self, frame: np.ndarray) -> tuple[list[Box], np.ndarray]:
        """Return the boxes, largest first, and the binary mask they came from."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if not self.min_aspect <= width / height <= self.max_aspect:
                continue
            boxes.append(Box(x, y, width, height))
        boxes.sort(key=lambda box: box.area, reverse=True)
        return boxes, mask

    def detect(self, frame: np.ndarray) -> list[Box]:
        return self.detect_with_mask(frame)[0]
