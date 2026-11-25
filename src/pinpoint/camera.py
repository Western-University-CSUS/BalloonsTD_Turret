"""Shared camera setup and FPS measurement."""

import argparse
import time
from collections.abc import Callable

import cv2


class CameraError(RuntimeError):
    """Raised when a camera or video source can't be opened or read."""


def parse_size(text: str) -> tuple[int, int]:
    """Parse a "WxH" string such as "1280x720" into (width, height)."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 1280x720, got {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 1280x720, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"width and height must be positive, got {text!r}")
    return width, height


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=None,
        metavar="WxH",
        help="requested resolution, e.g. 1280x720 (default: the camera's own default)",
    )


def open_camera(index: int = 0, size: tuple[int, int] | None = None) -> cv2.VideoCapture:
    """Open a camera, optionally requesting a resolution."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CameraError(f"could not open camera {index}")
    if size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise CameraError(f"could not open video {path}")
    return cap


def frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def measure_fps(
    cap: cv2.VideoCapture,
    frames: int = 30,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Time `frames` reads and return the real frames per second."""
    # Many cameras take noticeably longer to deliver the first frame.
    ok, _ = cap.read()
    if not ok:
        raise CameraError("failed to read a frame while measuring fps")
    start = clock()
    for _ in range(frames):
        ok, _ = cap.read()
        if not ok:
            raise CameraError("failed to read a frame while measuring fps")
    elapsed = clock() - start
    if elapsed <= 0:
        raise CameraError("measured zero elapsed time while measuring fps")
    return frames / elapsed


def describe_camera(cap: cv2.VideoCapture, measured_fps: float | None = None) -> str:
    """Return a one-line summary of resolution and fps."""
    width, height = frame_size(cap)
    text = f"{width}x{height}, driver reports {cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    if measured_fps is not None:
        text += f", measured {measured_fps:.1f} fps"
    return text


class FpsCounter:
    """Rolling FPS counter that updates once every `window` frames."""

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter):
        self.window = window
        self.clock = clock
        self.value: int | None = None
        self._count = 0
        self._start = clock()

    def tick(self) -> int | None:
        self._count += 1
        if self._count >= self.window:
            now = self.clock()
            elapsed = now - self._start
            if elapsed > 0:
                self.value = round(self._count / elapsed)
            self._count = 0
            self._start = now
        return self.value

    @property
    def text(self) -> str:
        return "Hold..." if self.value is None else str(self.value)
