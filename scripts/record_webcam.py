import argparse
import os
import sys
from datetime import datetime

import cv2

from pinpoint.camera import (
    CameraError,
    FpsCounter,
    add_camera_args,
    describe_camera,
    frame_size,
    measure_fps,
    open_camera,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record webcam footage to an mp4 file.")
    add_camera_args(parser)
    parser.add_argument("--out-dir", default="footage", help="output folder (default footage)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cap = None
    out = None
    try:
        cap = open_camera(args.camera, args.size)
        fps = measure_fps(cap)
        print(describe_camera(cap, fps))
        width, height = frame_size(cap)

        os.makedirs(args.out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(args.out_dir, f"testing_{stamp}.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            print(f"Could not open video writer for {path}", file=sys.stderr)
            return 1
        print(f"Recording to {path}")

        counter = FpsCounter()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera", file=sys.stderr)
                return 1
            out.write(frame)

            counter.tick()
            display = frame.copy()
            cv2.putText(
                display, counter.text, (width - 80, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0),
            )
            cv2.putText(
                display, "Press q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100),
            )
            cv2.imshow("Test", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    except CameraError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
