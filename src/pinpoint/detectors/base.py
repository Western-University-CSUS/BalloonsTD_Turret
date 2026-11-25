from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixels, origin at the top-left corner."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> int:
        return self.width * self.height


class Detector(Protocol):
    """Finds balloons in a BGR frame and returns their boxes, largest first."""

    def detect(self, frame: np.ndarray) -> list[Box]: ...
