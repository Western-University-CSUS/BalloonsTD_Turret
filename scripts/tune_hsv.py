import argparse
import sys

import cv2
import numpy as np

from pinpoint.camera import CameraError, add_camera_args, open_camera, open_video

CONTROLS = "Frame"
TRACKBARS = (
    ("H low", 0, 179),
    ("H high", 179, 179),
    ("S low", 0, 255),
    ("S high", 255, 255),
    ("V low", 0, 255),
    ("V high", 255, 255),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune one HSV range and press p to print it for COLOR_PRESETS. "
            "Colours that wrap hue 0/179, like red, need two runs."
        )
    )
    add_camera_args(parser)
    parser.add_argument("--video", metavar="PATH", help="loop a recorded video instead")
    return parser.parse_args(argv)


def read_range() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = (
        cv2.getTrackbarPos(name, CONTROLS) for name, _, _ in TRACKBARS
    )
    return (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cap = None
    try:
        cap = open_video(args.video) if args.video else open_camera(args.camera, args.size)
        cv2.namedWindow(CONTROLS)
        for name, start, top in TRACKBARS:
            cv2.createTrackbar(name, CONTROLS, start, top, lambda _: None)

        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Failed to read a frame from the camera", file=sys.stderr)
                return 1

            lower, upper = read_range()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            cv2.imshow(CONTROLS, frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Blend", cv2.bitwise_and(frame, frame, mask=mask))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print((lower, upper))
            elif key == ord("q"):
                return 0
    except CameraError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
