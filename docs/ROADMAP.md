# Pinpoint CV roadmap

What the CV team builds, in what order, and who owns it. Decided by Adam (CV lead) on 2026-11-25.
Words in **bold capitals** like Detection, Track, Session are defined in [GLOSSARY.md](../GLOSSARY.md).
The research behind each decision is in [docs/research/](research/).

- **Phase 1 (now): detection.** Two interchangeable detectors (classical HSV and YOLO26n), the data to train and test them, and a benchmark that compares them.
- **Phase 2 (later): tracking.** A hand-written Kalman tracker and target selection. Planned below, no issues yet.
- **Not ours:** launcher, gimbal motors and their control loop, arming and safety, operator UI, website, choice of turret computer.

## Milestones and tasks

Each task is one GitHub issue titled `T<n>: ...`. Every code task also needs `ruff check .` and `pytest` passing.

| # | Task | Owner | Milestone | Depends on |
|---|------|-------|-----------|------------|
| T1 | Shared detection types | @KevinHill14 | M1 Foundations | none |
| T2 | Recording fixes | @KevinHill14 | M1 Foundations | none |
| T4 | macOS camera-lock trial | @KevinHill14 | M1 Foundations | none |
| T5 | Public dataset prep | @asanghv2 | M1 Foundations | none |
| T8 | CVAT labelling project | @asanghv2 | M1 Foundations | none |
| T3 | Classical detector upgrade | @KevinHill14 | M2 Two detectors | T1 |
| T6 | Stage 1 training notebook | @asanghv2 | M2 Two detectors | T5 |
| T7 | `YoloDetector` | @asanghv2 | M2 Two detectors | T1, T6 |
| T9 | Frame extraction script | @asanghv2 | M2 Two detectors | T2 |
| T10 | Detection replay dump | @KevinHill14 | M2 Two detectors | T1, T2 |
| T11 | Benchmark script | @KevinHill14 | M3 Scoring | T1, T10 |
| T14 | Live-view script | @asanghv2 | M3 Scoring | T3, T7 |
| T12 | Own recording Sessions + split manifest | everyone (Adam owns the manifest) | later | T2, demo venue |
| T13 | Stage 2 retrain on own footage | @asanghv2 | later | T12, labels |

**Order.** Kevin does **T1 first**: it is small and T3, T7 and T10 wait on it. Aadi's M1 tasks (T5, T8) don't need T1, so both start on day one.

**Phase 1 is done when** both detectors run through T11 on our own val Sessions (a "dev" benchmark). The official benchmark waits for the frozen Test set (`test_v1`).

## Decisions this rests on

- **Detectors are interchangeable.** Both return `list[Detection]`. A Detection is a Box in float pixels of the full 1280×720 frame (origin top-left), a Score from 0 to 1, and a Colour class (`None` for YOLO). Detectors return every balloon, in no particular order; choosing a Target is not their job. Detectors are pure: no camera, no clock.
- **Capture time** is `time.monotonic()` in seconds, taken right after a frame is read. Recordings save it per frame.
- **Classical detector:** one HSV range set per colour class (red needs two ranges), median blur 5, open 5×5, close 9×9, then shape filters (area 600–10 000 px², circularity ≥ 0.6, aspect 0.6–1.3, solidity ≥ 0.85, enclosing-circle fill ≥ 0.5). Score = enclosing-circle fill. It can't see white, silver or foil balloons; YOLO covers those.
- **YOLO:** YOLO26n fine-tuned from `yolo26n.pt`, one class `balloon`, `imgsz=640`, `ultralytics` pinned. Deploy `nms=False` unless val shows it loses recall. Trained on Google Colab, data and checkpoints in the shared team Drive (never in git).
- **Data:** public data (Open Images V7 Balloon + serhiibiruk from Kaggle) goes in train only. Stage 1 trains on public data only. Stage 2 adds 8–12 of our own Sessions. Labelling in CVAT, owned by Aadi.
- **Split:** by whole Session, never by frame. 12 Sessions → 7 train / 2 val / 3 test (8 → 5 / 1 / 2). Test is demo-room footage, frozen and versioned (`test_v1` = placeholder room, `test_v2` = the real demo room). **Nothing is tuned on test, including HSV ranges.** Session id: `YYYYMMDD_<room>_<n>`.
- **Benchmark:** AP@0.5, plus precision/recall/F1 at each detector's Live threshold, median centre error (px), and `detect()` latency (median, p95). Reported for all balloons and for the classical-colour subset. Official runs only on a frozen Test set, appended to `docs/benchmarks.md`. Demo day uses the highest-AP detector that keeps up with the camera; the classical detector stays as the fallback.
- **Camera:** everyone develops on macOS, where OpenCV can't lock exposure or white balance. We've asked for a Linux turret computer (most likely a Raspberry Pi), where it can.

## Shared data layout (team Google Drive)

```
pinpoint/
  public/oi_balloon/       # T5, YOLO format
  public/serhiibiruk/      # T5, YOLO format, de-duplicated
  own/raw/<session_id>.mp4 + <session_id>_timestamps.csv   # T2/T12, every Session
  own/{train,val,test}/    # T9 frames + CVAT labels, split per docs/splits.csv
  runs/                    # T6/T13 training runs and weights
```

---

## M1 Foundations

### T1: Shared detection types

**Owner:** @KevinHill14. **Depends on:** nothing. **Blocks:** T3, T7, T10, T11.

**Why.** Both detectors and everything downstream (replay, benchmark, later the tracker) must speak one format. Today `Detector.detect()` returns integer `Box`es with no score or colour.

**Steps.**
1. In `src/pinpoint/detectors/base.py`: make `Box` fields floats, add `Detection`, and change `Detector` to return `list[Detection]`.
2. Add `src/pinpoint/frame.py` with `FrameResult`.
3. Update `HsvDetector` to return `Detection`s: `score=1.0` and `colour=None` for now (T3 sets real values). Drop the "largest first" sort.
4. Update `tests/test_hsv_detector.py` and any script that reads `.x`/`.y` etc.
5. Export the new names from `pinpoint.detectors`.

**Starter code.**
```python
@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels of the full frame, origin top-left."""
    x: float
    y: float
    width: float
    height: float
    # keep the existing center and area properties


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float               # 0..1; reject anything outside in __post_init__
    colour: str | None = None  # colour class name, None if the detector can't tell


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


# src/pinpoint/frame.py
@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    t_capture: float           # time.monotonic() seconds, taken right after the read
    detections: list[Detection]
    pose: tuple[float, float] | None = None   # gimbal (pan, tilt) deg; None until hardware exists
```

**Done when:** `Detection(box, 1.5)` raises `ValueError`; `HsvDetector().detect()` returns `Detection`s; all tests and scripts updated; ruff and pytest green.

### T2: Recording fixes

**Owner:** @KevinHill14. **Depends on:** nothing. **Blocks:** T9, T10, T12.

**Why.** Recorded Sessions are our val and test data and the tracker's replay input. Right now the FPS text is burned into saved frames, the writer assumes 30 fps, and there are no per-frame timestamps. Details: [classical-cv.md §7](research/classical-cv.md#7-recording-correctness).

**Steps** (in `scripts/record_webcam.py`):
1. Add `--session <id>` (format `YYYYMMDD_<room>_<n>`). Write `footage/<id>.mp4` and `footage/<id>_timestamps.csv`.
2. Draw the HUD on `display = frame.copy()` only. The saved frame stays raw.
3. Measure the real fps for about 2 s first (`camera.measure_fps`) and pass that to `VideoWriter`, not 30.
4. Write `frame_index,t_capture` per saved frame, `t_capture = time.monotonic()` taken right after `cap.read()` returns.
5. Request MJPG at 1280×720 (it works on Linux and Windows; macOS ignores it, which is fine):
   `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))` **before** setting width and height.
6. Move `out.write` to a writer thread fed by a bounded `queue.Queue(maxsize=60)`, so encoding never stalls capture. Stop cleanly: put a `None` sentinel, `join()` the thread, release the writer and the camera.

**Starter code.**
```python
def writer_loop(q: queue.Queue, out: cv2.VideoWriter, csv_file) -> None:
    while (item := q.get()) is not None:
        index, t_capture, frame = item
        out.write(frame)
        csv_file.write(f"{index},{t_capture:.6f}\n")
```

**Done when:** a 60 s recording has no overlay in the file; the CSV row count equals the video's frame count; the last `t_capture` minus the first is within 2 % of the video's frame count ÷ its fps.

### T4: macOS camera-lock trial

**Owner:** @KevinHill14. **Depends on:** nothing. No code.

**Why.** OpenCV can't lock exposure or white balance on macOS ([classical-cv.md §6](research/classical-cv.md#6-locking-c270-exposure-and-white-balance-per-backend)), so HSV ranges drift as lighting changes. A separate UVC-control tool might hold the setting.

**Steps.**
1. Try a macOS UVC control tool on the C270, for example the open-source `uvc-util` or a "webcam settings" app from the App Store. None of them is verified with the C270 yet.
2. Lock exposure and white balance, then start `scripts/record_webcam.py`. **Check the lock survives OpenCV opening the camera.**
3. Test: record 30 s pointing at a fixed scene, then wave a bright white sheet in and out of frame. Compare the mean V of a fixed background patch with and without the lock.

**Done when:** the issue has a short report: tool name and version, whether the lock holds after OpenCV opens the camera, and the before/after V numbers. If it works, add the setup steps to `docs/camera-setup.md`.

### T5: Public dataset prep

**Owner:** @asanghv2. **Depends on:** nothing. **Blocks:** T6.

**Why.** Stage 1 trains only on public data to prove the training pipeline. Details: [datasets-and-split.md §1](research/datasets-and-split.md#1-dataset-by-dataset-evaluation).

**Steps.**
1. **Open Images V7 Balloon** via FiftyOne (`pip install fiftyone`). Export the `train` split, and the `validation` split separately (it becomes Stage 1 val). **Drop group-of boxes** (one box around a whole bunch). Don't use the Ultralytics `open-images-v7.yaml` auto-download: it pulls about 561 GB.
2. **serhiibiruk** (Kaggle `serhiibiruk/balloon-object-detection`), already YOLO format. Use **`train/` only, never its `valid/`** (it leaks into train). De-duplicate Roboflow copies by the filename stem before `_jpg.rf.`: keep one file per stem. Expect roughly 500 originals.
3. Make every label class id `0` (`balloon`).
4. Upload to the Drive layout above. Put the conversion code in `scripts/prepare_public_data.py` so it can be re-run.

**Starter code** (from the research; check the label field name with `ds.get_field_schema()`):
```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

for split, out_split in (("train", "train"), ("validation", "val")):
    ds = foz.load_zoo_dataset("open-images-v7", split=split,
                              label_types=["detections"], classes=["Balloon"])
    view = ds.filter_labels("ground_truth",
                            (F("label") == "Balloon") & (F("IsGroupOf") == False))  # noqa: E712
    view.export(export_dir="oi_balloon", dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth", split=out_split, classes=["Balloon"])
```

**Done when:** both sets are in the Drive; the issue lists image and box counts per set (Open Images train should be about 2.3k images after dropping group-of boxes); 20 random images per set checked by eye with their boxes drawn.

### T8: CVAT labelling project

**Owner:** @asanghv2. **Depends on:** nothing. **Blocks:** labelling for T11, T12, T13.

**Steps.**
1. Create a project on the free cloud CVAT (app.cvat.ai). Invite Kevin and Adam.
2. One label, `balloon`, with a **required `colour` attribute** (select): red, orange, yellow, green, blue, purple, pink, white, silver/foil, other.
3. **One CVAT task per Session**, named with the session id, so splits stay per Session.
4. Write a short labelling guide in the project description: label **every** balloon, tight boxes, ignore strings, and frames with no balloon stay empty (that is allowed).
5. Check both export formats on a 5-frame test task:
   - **YOLO 1.1** for training. It drops attributes.
   - **CVAT for images 1.1** (XML) for val and test, which keeps `colour` for the benchmark (T11).

**Done when:** the project exists with the label and attribute; both exports from the test task are in the Drive; the guide is written.

---

## M2 Two detectors

### T3: Classical detector upgrade

**Owner:** @KevinHill14. **Depends on:** T1.

**Why.** Today the detector uses one colour range, no cleanup and almost no shape filtering. The recipe below is in [classical-cv.md](research/classical-cv.md) (§1–3 and "Recommended classical detector recipe").

**Steps.**
1. **Settings file:** `config/hsv_ranges.json`, `{ "red": [[[0,100,60],[10,255,255]], [[170,100,60],[179,255,255]]], "blue": [...] }`. Start from the research table; these are starting points, not calibrated values. Load it in `HsvDetector.from_file(path)`.
2. **`scripts/tune_hsv.py`:** add `--colour <name> --save` to write the tuned range into the settings file. Red needs two ranges, so `--range-index 0|1`. **Tune on train or val footage only, never test.**
3. **Per colour class:** OR that class's ranges → open 5×5 → close 9×9 (elliptical kernels) → `findContours(RETR_EXTERNAL)`. Run `medianBlur(5)` once before the HSV conversion.
4. **Filters per contour** (all starting values, kept as constructor arguments):

| Filter | How | Keep if |
|---|---|---|
| area | `cv2.contourArea(c)` | 600–10 000 px² |
| circularity | `4*pi*A / arcLength(c, True)**2` | ≥ 0.6 |
| aspect | `w / h` from `boundingRect` | 0.6–1.3 |
| solidity | `A / contourArea(convexHull(c))` | ≥ 0.85 |
| fill | `A / (pi * r**2)`, `(_, r) = minEnclosingCircle(c)` | ≥ 0.5 |

5. Return **every** survivor as `Detection(Box(x, y, w, h), score=fill, colour=class_name)`, with `score` clamped to at most 1.0.

**Starter code.**
```python
K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

hsv = cv2.cvtColor(cv2.medianBlur(image, 5), cv2.COLOR_BGR2HSV)
for name, ranges in self.classes.items():
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter each contour (table above) and append a Detection
```

**Tests.** The existing tests paint rectangles, which the circularity filter will now reject. Paint filled circles or ellipses with `cv2.circle` / `cv2.ellipse` instead. Cover:
- red found in both hue ranges
- a red and a blue balloon in one frame → 2 Detections with the right colours
- each filter rejects its bad shape (e.g. a long thin bar fails aspect)
- score is in 0..1

**Done when:** tests above pass; on a recorded Session with at least two colours, `scripts/live.py` (T14) or a quick script shows every balloon found.

### T6: Stage 1 training notebook

**Owner:** @asanghv2. **Depends on:** T5. **Blocks:** T7.

**Why.** Stage 1 proves the Colab → Drive → weights pipeline. Its scores don't count as results, because val is Open Images, not our camera. Details: [yolo26.md §9](research/yolo26.md#9-colab-training).

**Steps.**
1. `notebooks/train_stage1.ipynb` in the repo. **Clear outputs before committing.**
2. `balloons_stage1.yaml` in the Drive:
   ```yaml
   path: /content/drive/MyDrive/pinpoint
   train: [public/oi_balloon/images/train, public/serhiibiruk/images]
   val: public/oi_balloon/images/val
   names: {0: balloon}
   ```
3. Train, then validate both heads, then copy `best.pt` to `runs/stage1/` in the Drive.

**Starter code.**
```python
!pip install "ultralytics==8.4.218"   # pinned: the meaning of nms changed in 8.4.142
from google.colab import drive; drive.mount("/content/drive")
from ultralytics import YOLO

model = YOLO("yolo26n.pt")             # COCO-pretrained; fine-tune, don't train from scratch
model.train(data="/content/drive/MyDrive/pinpoint/balloons_stage1.yaml",
            epochs=100, imgsz=640, batch=-1,
            project="/content/drive/MyDrive/pinpoint/runs", name="stage1", save_period=10)
# After a disconnect: YOLO(".../runs/stage1/weights/last.pt").train(resume=True)

best = YOLO("/content/drive/MyDrive/pinpoint/runs/stage1/weights/best.pt")
print(best.val().box.map50, best.val(nms=False).box.map50)   # default NMS head vs NMS-free head
```

**Done when:** the notebook runs top to bottom on a fresh Colab session; `best.pt` is in the Drive; the issue records the mAP50 with the default NMS head and with `nms=False` and the training time.

### T7: `YoloDetector`

**Owner:** @asanghv2. **Depends on:** T1, T6.

**Steps.**
1. In `pyproject.toml`, add an optional extra `yolo = ["ultralytics==8.4.218"]`, installed with `pip install -e ".[dev,yolo]"`. CI doesn't install it.
2. `src/pinpoint/detectors/yolo.py`: import `ultralytics` **inside** `__init__`, so `pinpoint.detectors` still imports without it.
3. Convert the results to `Detection`s with `colour=None`. Boxes come back in original-frame pixels already.
4. Weights are a constructor argument (a path in the Drive). Never commit `.pt` files.

**Starter code.**
```python
class YoloDetector:
    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, nms: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.nms = imgsz, conf, nms

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                                    nms=self.nms, verbose=False)[0]
        # result.boxes.xyxy (N x 4) and result.boxes.conf (N) -> Detection(Box(x1, y1, x2 - x1, y2 - y1), score, None)
```

**Tests:** `pytest.importorskip("ultralytics")`, then check `detect()` on a blank 1280×720 image returns a list of `Detection` (likely empty). These tests are skipped in CI.

**Done when:** the Stage 1 weights find balloons on a recorded Session (checked with T14 or a short script); tests pass locally with ultralytics installed and are skipped without it.

### T9: Frame extraction script

**Owner:** @asanghv2. **Depends on:** T2.

**Why.** Labelling every frame of a 30 fps video wastes effort on near-duplicates. We label 1–2 frames per second.

**Steps.**
1. `scripts/extract_frames.py --video footage/<id>.mp4 --out <dir> --fps 2`.
2. Keep a frame when at least `1/fps` seconds of `t_capture` (from the T2 CSV) have passed since the last kept frame **and** it differs enough from the last kept frame: mean absolute pixel difference on a 160×90 grey thumbnail above a threshold (`--min-diff`, start at 4).
3. Name files `<session_id>_<frame_index:06d>.jpg`, so every frame traces back to its Session and its `t_capture`.

**Done when:** a unit test covers the keep/skip rule on synthetic frames; running it on a 60 s Session gives roughly 60–120 frames.

### T10: Detection replay dump

**Owner:** @KevinHill14. **Depends on:** T1, T2. **Blocks:** T11, and the tracker in Phase 2.

**Why.** Running a detector once over a recording and saving the result makes the benchmark and the later tracker fast and repeatable, with no camera needed.

**Steps.**
1. `scripts/dump_detections.py --video footage/<id>.mp4 --detector hsv|yolo [--weights PATH] --out <id>_<detector>.jsonl`. It reads `t_capture` from the T2 CSV.
2. One JSON line per frame, a FrameResult:
   `{"frame_id": 12, "t_capture": 1234.567890, "detections": [{"x": 101.5, "y": 40.0, "width": 52.0, "height": 61.0, "score": 0.83, "colour": "red"}]}`
3. Add `to_dict()`/`from_dict()` on `FrameResult` (in `src/pinpoint/frame.py`), plus a `read_jsonl(path) -> Iterator[FrameResult]` helper.

**Done when:** a round-trip test passes (`FrameResult` → JSON line → `FrameResult` gives an equal object); dumping a 60 s Session gives one line per video frame.

---

## M3 Scoring

### T11: Benchmark script

**Owner:** @KevinHill14. **Depends on:** T1, T10, plus labelled val frames (T8).

**Why.** This is how we compare the two detectors, and how the demo-day detector gets chosen.

**Steps.**
1. `benchmarks/run.py --detector hsv|yolo [--weights] --frames <dir> --labels <cvat.xml> --live-threshold 0.5 [--official --test-version test_v1]`.
2. **Matching:** for each frame, sort detections by score; each one matches the unmatched ground-truth box with the highest IoU if IoU ≥ 0.5, otherwise it's a false positive.
3. **AP@0.5:** area under the precision–recall curve over all frames (all-point interpolation). Also report precision, recall and F1 at `--live-threshold`, and the **median centre error in px** of matched pairs.
4. **Latency:** `time.perf_counter()` around `detect()` only, after 10 warm-up frames; report median and p95 in ms, plus the machine name (`platform.node()`, `platform.processor()`).
5. **Two rows:** all balloons, and the classical-colour subset. For the subset, ground-truth boxes whose `colour` attribute isn't in `config/hsv_ranges.json` become "ignore" boxes: a detection matching one counts as neither a true positive nor a false positive.
6. Print a table plus metadata: git commit (`git rev-parse --short HEAD`), the weights file's SHA-256, and the HSV settings file's SHA-256.
7. `--official` appends one row to `docs/benchmarks.md`. **Only on a frozen Test set**; val runs are dev runs and aren't recorded.

**Tests:** a hand-computed AP on a tiny 3-frame case; the ignore-box rule; the IoU function.

**Done when:** tests pass; a dev run of both detectors on our val Sessions prints both rows. **This is the Phase 1 exit check.**

### T14: Live-view script

**Owner:** @asanghv2. **Depends on:** T3, T7.

**Steps.**
1. `scripts/live.py --detector hsv|yolo [--weights PATH] [--video PATH]` (camera by default; reuse `add_camera_args`).
2. Draw boxes, score and colour on a **display copy only**, plus an fps counter (`camera.FpsCounter`).
3. Delete `scripts/hsv_testing.py`, which this replaces.

**Done when:** both detectors run live on the webcam and on a recorded Session; `hsv_testing.py` is gone.

---

## Later (no issues yet)

### T12: Own recording Sessions and split manifest

Everyone records; Adam owns `docs/splits.csv` (`session_id,split,test_version,room,notes`).

- 8–12 Sessions: different rooms and lighting, balloons at 3–6 m, at least 4 colours, a still and a moving camera, and 5–10 % of frames with no balloon.
- Record with T2; extract with T9; label in CVAT (T8), one task per Session.
- Val Sessions come from rooms that are **not** the demo room. Test Sessions come from the demo room. Until we can get in there, a placeholder room becomes `test_v1`.
- Keep the raw video of **every** Session in the Drive.

Waits on: the demo venue.

### T13: Stage 2 retrain

Retrain from the Stage 1 weights with our own train Sessions added; val switches to our own val Sessions. Then re-run T11. Waits on: T12 and labels.

---

## Phase 2: tracking (planned, not filed)

Built after Phase 1, all testable offline with recorded Sessions (gimbal pose = (0, 0), static camera). Details: [kalman-pid.md](research/kalman-pid.md) §1–3.

1. **Pixel ↔ angle utilities.** Start from the 55° diagonal prior (f ≈ 1411 px at 1280×720). Replace it with a chessboard `cv2.calibrateCamera` result stored in a settings file. One function, including the image-y flip, with a round-trip test.
2. **Replay format.** Already done by T10.
3. **Hand-written Kalman filter:** constant velocity, one 2-state filter (angle, rate) per axis, base-frame degrees, `dt` from Capture time. `filterpy` only as a test oracle. `R` measured per detector from a stationary balloon; `Q` tuned until NIS ≈ 1 on replay.
4. **Track manager:** Mahalanobis gate (chi-square 9.21), brute-force exact assignment (6 balloons or fewer). Confirm after 3 hits in 5 frames, coast up to 1 s, then delete.
5. **Target selection:** only confirmed Tracks are eligible. Stay on the Target until its Track is deleted, then take the confirmed Track nearest the current aim. Optional colour filter.
6. **Synthetic gimbal-motion test:** shift detections by a fake pan/tilt and check static balloons stay static in base-frame angles.

## Not decided yet

- **Demo venue:** when we can record there, and with which balloons. Needed for the Test set.
- **Turret computer speed:** YOLO26n at 640 on a Pi CPU is probably under 30 fps. Benchmark on a Pi early, then choose NCNN or ONNX, detection every Nth frame, or a smaller imgsz.
- **Colour for YOLO Detections:** a future colour-targeting feature needs the colour read from the box.
- **Aim corrections and fire decision** (parallax, dart drop, aim-ahead): need dart speed, barrel offset and latency from the launcher team.
- **Gimbal control loop:** probably the gimbal team's; to be confirmed with them.
- **Licence:** an AGPL-3.0 `LICENSE` (required by `ultralytics`) gets added before the project ends.
