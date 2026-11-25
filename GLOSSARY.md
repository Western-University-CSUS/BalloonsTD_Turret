# Pinpoint

The CV pipeline of a turret that finds, tracks and aims at helium balloons. Camera → Detector → Tracker → Controller.

## Detection

**Detector**:
A component that finds balloons in one image. Classical (HSV colour) and YOLO detectors are interchangeable.
_Avoid_: model, finder

**Detection**:
One balloon found in one frame: its box, its score, and its colour class if known.
_Avoid_: hit, blob, result

**Box**:
The axis-aligned rectangle around a balloon, in pixels of the full capture frame, origin top-left.
_Avoid_: bbox, rect

**Score**:
A detector's confidence in a Detection, from 0 to 1.
_Avoid_: confidence, probability

**Colour class**:
A named balloon colour (red, blue, ...) defined by HSV ranges. YOLO detections have none.
_Avoid_: color label, hue

**Frame result**:
Every Detection from one frame, together with the frame's id, capture time and gimbal pose.
_Avoid_: frame detections, batch

**Capture time**:
The monotonic-clock time at which a frame was read from the camera.
_Avoid_: timestamp (unqualified), frame time

## Tracking and aiming

**Base-frame angles**:
Pan and tilt in degrees relative to the turret's fixed base, pan positive right, tilt positive up, zero at the gimbal's home.
_Avoid_: world coordinates, camera angles

**Gimbal pose**:
The gimbal's pan and tilt at a given capture time.
_Avoid_: position, orientation

**Track**:
One balloon followed over time, with a filtered base-frame position, rate and uncertainty.
_Avoid_: target, object

**Track status**:
Tentative (newly seen), confirmed (seen enough to trust), or coasting (temporarily unseen).

**Target**:
The one Track the turret is currently aiming at.
_Avoid_: track (when meaning the chosen one)

**Command**:
The per-frame instruction to the gimbal and launcher: final barrel angles, feedforward rates, a fire request, and the Target's track id or none.
_Avoid_: aim, order, message

## Data

**Session**:
One continuous recording of balloons in one room under one setup; the unit of the train/val/test split.
_Avoid_: clip, run, video

**Test set**:
The frozen, versioned set of Sessions used only for the Benchmark, never for training or tuning.
_Avoid_: holdout, eval set

**Benchmark**:
The head-to-head scoring of Detectors on the Test set. Only runs on a frozen Test set are official.
_Avoid_: eval, comparison

**Live threshold**:
The Score below which a Detector's Detections are discarded when it runs on the turret.
_Avoid_: cutoff, confidence threshold
