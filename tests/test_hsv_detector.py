import cv2
import numpy as np
import pytest

from pinpoint.detectors import COLOR_PRESETS, Box, HsvDetector

CYAN = (92, 200, 200)
RED_LOW = (5, 200, 200)
RED_HIGH = (175, 200, 200)
GREEN = (60, 200, 200)


def blank_frame(width=320, height=240):
    return np.zeros((height, width, 3), dtype=np.uint8)


def paint(frame, x, y, width, height, hsv):
    pixel = np.uint8([[hsv]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    frame[y : y + height, x : x + width] = bgr
    return frame


def test_finds_cyan_rectangle_with_exact_box():
    frame = paint(blank_frame(), 50, 40, 60, 50, CYAN)
    assert HsvDetector.from_preset("cyan").detect(frame) == [Box(50, 40, 60, 50)]


def test_empty_frame_gives_nothing():
    assert HsvDetector.from_preset("cyan").detect(blank_frame()) == []


def test_wrong_colour_is_ignored():
    frame = paint(blank_frame(), 50, 40, 60, 50, GREEN)
    assert HsvDetector.from_preset("cyan").detect(frame) == []


def test_small_blob_is_filtered():
    frame = paint(blank_frame(), 50, 40, 10, 10, CYAN)
    assert HsvDetector.from_preset("cyan", min_area=500).detect(frame) == []


def test_small_blob_cannot_beat_real_balloon():
    frame = blank_frame()
    paint(frame, 10, 10, 8, 8, CYAN)
    paint(frame, 150, 100, 50, 60, CYAN)
    assert HsvDetector.from_preset("cyan").detect(frame) == [Box(150, 100, 50, 60)]


def test_wide_blob_is_rejected_by_aspect_ratio():
    frame = paint(blank_frame(), 10, 100, 200, 20, CYAN)
    assert HsvDetector.from_preset("cyan").detect(frame) == []


def test_red_is_found_at_both_ends_of_hue_axis():
    frame = blank_frame()
    paint(frame, 20, 20, 50, 50, RED_LOW)
    paint(frame, 200, 120, 40, 40, RED_HIGH)
    boxes = HsvDetector.from_preset("red").detect(frame)
    assert boxes == [Box(20, 20, 50, 50), Box(200, 120, 40, 40)]


def test_boxes_are_sorted_largest_first():
    frame = blank_frame()
    paint(frame, 10, 10, 30, 30, CYAN)
    paint(frame, 100, 10, 60, 60, CYAN)
    paint(frame, 200, 150, 40, 40, CYAN)
    areas = [box.area for box in HsvDetector.from_preset("cyan").detect(frame)]
    assert areas == [3600, 1600, 900]


def test_detect_with_mask_returns_binary_mask():
    frame = paint(blank_frame(), 50, 40, 60, 50, CYAN)
    boxes, mask = HsvDetector.from_preset("cyan").detect_with_mask(frame)
    assert boxes == [Box(50, 40, 60, 50)]
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 60 * 50


def test_box_center_and_area():
    box = Box(10, 20, 30, 40)
    assert box.center == (25.0, 40.0)
    assert box.area == 1200


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        HsvDetector.from_preset("purple")


@pytest.mark.parametrize(
    "ranges",
    [
        [((10, 0, 0), (5, 255, 255))],
        [((0, 0, 0), (180, 255, 255))],
        [((0, 0, 0), (179, 256, 255))],
        [((-1, 0, 0), (179, 255, 255))],
        [((0, 0), (179, 255))],
    ],
)
def test_bad_ranges_raise(ranges):
    with pytest.raises(ValueError):
        HsvDetector(ranges)


def test_empty_ranges_raise():
    with pytest.raises(ValueError):
        HsvDetector([])


@pytest.mark.parametrize(("min_aspect", "max_aspect"), [(0, 2.0), (2.0, 1.0), (-1, 1)])
def test_bad_aspect_limits_raise(min_aspect, max_aspect):
    with pytest.raises(ValueError):
        HsvDetector(COLOR_PRESETS["cyan"], min_aspect=min_aspect, max_aspect=max_aspect)


@pytest.mark.parametrize("name", sorted(COLOR_PRESETS))
def test_every_preset_builds(name):
    assert isinstance(HsvDetector.from_preset(name), HsvDetector)
