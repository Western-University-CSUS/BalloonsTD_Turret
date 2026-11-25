import argparse

import numpy as np
import pytest

from pinpoint.camera import CameraError, FpsCounter, measure_fps, parse_size


class FakeCapture:
    def __init__(self, reads_ok: int):
        self.reads_ok = reads_ok
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads > self.reads_ok:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class FakeClock:
    def __init__(self, times):
        self.times = iter(times)

    def __call__(self):
        return next(self.times)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1280x720", (1280, 720)), ("640X480", (640, 480)), ("1x1", (1, 1))],
)
def test_parse_size_accepts_wxh(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "1280", "1280x", "x720", "axb", "0x720", "-1x720", "1x2x3"])
def test_parse_size_rejects_bad_input(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size(text)


def test_measure_fps_skips_first_frame_and_times_the_rest():
    cap = FakeCapture(reads_ok=100)
    fps = measure_fps(cap, frames=30, clock=FakeClock([10.0, 11.5]))
    assert fps == pytest.approx(20.0)
    assert cap.reads == 31


def test_measure_fps_raises_on_failed_first_read():
    with pytest.raises(CameraError):
        measure_fps(FakeCapture(reads_ok=0), clock=FakeClock([0.0, 1.0]))


def test_measure_fps_raises_on_failed_later_read():
    with pytest.raises(CameraError):
        measure_fps(FakeCapture(reads_ok=5), frames=30, clock=FakeClock([0.0, 1.0]))


def test_measure_fps_raises_on_zero_elapsed():
    with pytest.raises(CameraError):
        measure_fps(FakeCapture(reads_ok=100), frames=30, clock=FakeClock([5.0, 5.0]))


def test_fps_counter_holds_until_first_window_then_reports():
    clock = FakeClock([0.0, 1.0, 1.5])
    counter = FpsCounter(window=3, clock=clock)
    assert counter.text == "Hold..."
    assert counter.tick() is None
    assert counter.tick() is None
    assert counter.text == "Hold..."
    assert counter.tick() == 3
    assert counter.value == 3
    assert counter.text == "3"
    counter.tick()
    counter.tick()
    assert counter.tick() == 6
