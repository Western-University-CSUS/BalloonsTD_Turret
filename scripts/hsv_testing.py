import argparse
import sys

import cv2

from pinpoint.camera import (
    CameraError,
    FpsCounter,
    add_camera_args,
    describe_camera,
    frame_size,
    open_camera,
    open_video,
)
from pinpoint.detectors import COLOR_PRESETS, HsvDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show HSV detection on a camera or video.")
    add_camera_args(parser)
    parser.add_argument("--video", metavar="PATH", help="use a recorded video instead")
    parser.add_argument(
        "--color", choices=sorted(COLOR_PRESETS), default="cyan", help="colour preset"
    )
    parser.add_argument(
        "--min-area", type=float, default=500.0, help="smallest blob area in pixels"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detector = HsvDetector.from_preset(args.color, min_area=args.min_area)
    cap = None
    try:
        cap = open_video(args.video) if args.video else open_camera(args.camera, args.size)
        print(describe_camera(cap))
        width, height = frame_size(cap)

        counter = FpsCounter()
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    return 0
                print("Failed to read a frame from the camera", file=sys.stderr)
                return 1

            boxes, mask = detector.detect_with_mask(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blend = cv2.bitwise_and(frame, frame, mask=mask)
            for box in boxes:
                cv2.rectangle(
                    frame, (box.x, box.y), (box.x + box.width, box.y + box.height),
                    (0, 255, 0), 2,
                )

            counter.tick()
            cv2.putText(
                frame, counter.text, (width - 80, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0),
            )
            cv2.putText(
                frame, "Press q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100),
            )
            cv2.imshow("BGR", frame)
            cv2.imshow("HSV", hsv)
            cv2.imshow("Mask", mask)
            cv2.imshow("Blend", blend)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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
