# Classical CV for balloon detection: best practice vs. what we have

> [!NOTE]
> **Status (2026-11-25):** this is the research the classical detector decision was based on. Most of it was adopted. Where the team decided differently, **[ROADMAP.md](../ROADMAP.md) wins**:
> - **Output format:** detectors return `Detection(box, score, colour)` (see `GLOSSARY.md`), not the `(name, cx, cy, r, area)` tuples in the recipe below. The **score is the enclosing-circle fill** (`A / (pi r²)`).
> - **Tracker:** association is by Mahalanobis distance in angle space, not "nearest centre" as mentioned at the end.
> - **HoughCircles:** left out for now. Revisit only if clutter false positives remain after the filters.
> - **Camera lock on macOS:** we trial a webcam-settings app (T4); otherwise we use wider ranges. We've asked for the turret computer to be Linux.

Researched 2026-11-25. Scope: Logitech C270 (720p, fixed focus), helium balloons 3-6 m away indoors,
roughly 40-65 px wide in a 1280x720 frame, colours unknown, several balloons in view.

**Source notes.** OpenCV source is cited at a pinned commit of the `4.x` branch,
`4a401fd` (`OCV` below = `https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/`).
docs.opencv.org refuses automated fetches, so OpenCV documentation is cited from the files it is
generated from in that same repo (the text is identical). Each citation is marked *[source]*,
*[docs]*, *[issue]*, *[spec]* or *[secondary]*. Anything I could not confirm against a primary
source is marked **unverified**.

---

## 1. HSV thresholding for several colours

**How OpenCV's HSV works.** For 8-bit images `cvtColor(..., COLOR_BGR2HSV)` stores `H <- H/2` "to fit
to 0 to 255", so hue is 0-179 and S, V are 0-255
([docs: colors.markdown L100](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/imgproc/doc/colors.markdown);
[docs: py_colorspaces L29](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.markdown):
"For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]").
`COLOR_BGR2HSV_FULL` uses the full 0-255 range for hue instead. Pick one and never mix them.

**Red wraps around.** Red sits at both ends of the hue circle, so a single `inRange` cannot cover it.
Use two masks and OR them:

```python
red = cv2.inRange(hsv, (0, S_MIN, V_MIN), (10, 255, 255)) | \
      cv2.inRange(hsv, (170, S_MIN, V_MIN), (179, 255, 255))
```

(The `inRange` + bitwise pattern is the one in the official tutorial,
[docs: py_colorspaces L42-65](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.markdown).)
An alternative with no wrap special case: shift hue by 90 (`(h + 90) % 180`) for the red class only.

**Several colours at once.** Best practice is a table of named classes, each a list of
`(lower, upper)` ranges (red has two), each run through `inRange` on the same HSV frame, and each
class's mask processed and contoured separately. Running classes separately lets two touching
balloons of different colours stay separate, and lets each detection carry a colour label.
Starting hue bands (OpenCV 0-179 scale; **these are starting points only, not sourced; calibrate them on the
real balloons under locked exposure**):

| Class | H ranges | Notes |
|---|---|---|
| red | 0-10, 170-179 | two ranges |
| orange | 10-22 | overlaps skin tones (see section 4) |
| yellow | 22-35 | |
| green | 35-85 | |
| cyan/blue | 85-130 | Kevin's cyan 85-100 sits here |
| purple | 130-160 | |
| pink/magenta | 160-170 | pastel pink has low S, see section 4 |

**Calibration tool.** Ranges must be measured, not guessed: a trackbar tool (`cv2.createTrackbar`)
for H/S/V min/max per class, saved to a JSON/YAML file the detector loads. Calibrate with exposure
and white balance locked (section 6); with auto exposure on, V and S drift as the scene changes, so
the calibrated range stops matching.

## 2. Mask cleanup: blur, open, close

- **Blur before HSV conversion** (`GaussianBlur` 5x5 or `medianBlur` 5). This removes sensor noise
  and JPEG block artefacts that make salt-and-pepper speckle in the mask. Median is better at keeping
  the balloon edge.
- **Open (erode then dilate)** removes small white specks: "useful in removing noise"
  ([docs: py_morphological_ops L64-69](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.markdown)).
- **Close (dilate then erode)** fills small black holes inside the object
  ([same file, L75 onward](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.markdown)).
  For balloons this matters: the **specular highlight** is a white, low-saturation spot that falls
  outside the colour range and punches a hole in the mask (section 4).
- Use an elliptical kernel (`getStructuringElement(MORPH_ELLIPSE, ...)`), sized against the target:
  a balloon here is about 40-65 px wide, so open with 3x3 to 5x5 (kills specks of a few px) and
  close with 7x7 to 9x9 (fills a highlight of roughly 5-15 px). A kernel bigger than about 1/4 of
  the balloon width starts merging neighbours.

## 3. Contour filtering, and returning every balloon

`findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` is right: `RETR_EXTERNAL` drops the inner
contours (holes) that Kevin's notes mention (`docs/classic_CV_research.md:18`). Then filter **every**
contour instead of taking `max(contours, key=contourArea)`:

| Filter | Formula / API | Suggested start for this rig |
|---|---|---|
| Area | `contourArea` | balloon about 40-65 px wide, oval about 1.2x taller: about 1500-4000 px². Accept about 600-10000 to allow partial occlusion and the 3 m end. |
| Circularity | `4*pi*A / P²` with `P = arcLength(c, True)` | >= 0.6 (a perfect circle is 1.0; an oval balloon is about 0.8-0.95) |
| Aspect ratio | `w/h` from `boundingRect` ([docs: py_contour_properties L14](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.markdown)) | 0.6-1.3 (balloons are taller than wide; the string adds nothing to a colour mask) |
| Solidity | `A / contourArea(convexHull(c))` ([same file L38-48](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.markdown)) | >= 0.85 (rejects ragged background blobs) |
| Fill of enclosing circle | `A / (pi r²)`, `(x,y),r = minEnclosingCircle(c)` ([docs: py_contour_features L168-171](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.markdown)) | >= 0.5 |

Circularity, solidity and fill thresholds are engineering starting points (**not sourced**); tune
them on recorded footage. For aiming, `minEnclosingCircle` gives a centre and radius directly; the
radius also gives a rough range estimate (pinhole model). The contour centroid from `moments`
(`cx = m10/m00`) is more stable than the bounding-box centre when part of the balloon is clipped.

Return a list of detections `(class, cx, cy, r, area, score)` for **all** survivors, sorted by
a score if needed. Target selection belongs in the tracker/turret logic, not in the detector.

## 4. Specular highlights, low saturation, similar backgrounds

- **Specular highlights.** Latex and foil balloons reflect ceiling lights as near-white spots
  (S low, V near 255). They fall outside the colour range, so they leave a hole or a notch in the
  mask. Fixes: morphological close (section 2); `RETR_EXTERNAL` so a hole never splits the contour;
  measure on the filled contour. For foil/mylar balloons, colour thresholding is unreliable; plan
  for a learned detector there.
- **Low saturation.** Kevin noted this correctly (`docs/classic_CV_research.md:6`). Near the grey
  axis, hue is numerically unstable, so set `S_MIN` around 80-120 and `V_MIN` around 50-70 for every
  class. Consequence: **white, black, silver and pastel balloons cannot be found by hue at all**.
  That is a real limit of this approach; say so to the team.
- **Similar-coloured backgrounds** (skin for orange/red, wood, posters, clothing). Mitigations, in
  order of cost: (1) shape filters (section 3) reject most of them; (2) a fixed region-of-interest
  mask for known background clutter; (3) temporal persistence: require a detection in N of the last
  M frames, which a tracker gives for free; (4) background subtraction
  (`createBackgroundSubtractorMOG2`) ANDed with the colour mask, since helium balloons move and
  most walls do not (fails once the turret pans, since the whole scene then moves).

## 5. Alternatives to compare

- **HoughCircles.** Works on a grayscale gradient image, not on colour. The OpenCV doc warns
  "it may fail to find correct radii. You can assist to the function by specifying the radius range
  (minRadius and maxRadius)" ([docs: imgproc.hpp L2287-2289](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/imgproc/include/opencv2/imgproc.hpp)).
  `HOUGH_GRADIENT_ALT` is "a variation of HOUGH_GRADIENT to get better accuracy" and its `param2`
  is a "perfectness" measure ([imgproc.hpp L492, L2306-2313](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/imgproc/include/opencv2/imgproc.hpp)).
  Balloons are ovals with soft, low-contrast edges against walls, and Hough fires on any round
  texture, so it is a poor primary detector here. Its good use is **verification inside a colour
  ROI**: run it on the cropped grayscale patch with `minRadius` about 15 and `maxRadius` about 40 px.
  It is colour-agnostic, so it is the one classical route to white/grey balloons, at the cost of many
  false positives.
- **Adaptive colour.** (a) Histogram back-projection (`calcHist` on a seed patch, then
  `calcBackProject`) learns each balloon's actual colour distribution once seen, instead of a fixed
  box in HSV; it pairs naturally with CamShift for tracking. (b) Per-frame S/V normalisation
  (e.g. CLAHE on V) helps when lighting is uneven, but it changes V, so calibrate after it, not
  before. (c) Lab colour space with a/b distance to a reference colour is less sensitive to
  brightness than HSV boxes. None of these fix an unlocked camera; lock exposure first.
- **Learned detector (YOLO).** Out of scope here, but it is the escape hatch for white/foil balloons
  and cluttered rooms. Kevin's choice of axis-aligned boxes (`docs/classic_CV_research.md:20`)
  keeps classical output compatible with it, which is sensible.

## 6. Locking C270 exposure and white balance, per backend

Why it matters: with auto exposure and auto white balance on, the camera changes gain and colour
balance as balloons and people move through the scene, so S and V shift and the calibrated ranges
drift. Property support in OpenCV is backend-specific; a maintainer: "Almost all videoio properties
are optional and may have backend-specific implementation"
([issue #19621](https://github.com/opencv/opencv/issues/19621)). Always check the return value of
`cap.set(...)` and read the value back with `cap.get(...)`.

The OpenCV header's own doc for `CAP_PROP_AUTO_EXPOSURE` only says "DC1394: exposure control done by
camera..." ([videoio.hpp L162](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/include/opencv2/videoio.hpp)), i.e. the value
meaning is undocumented and differs per backend, as below.

### macOS, AVFoundation (`CAP_AVFOUNDATION`): not possible from OpenCV

- `CvCaptureCAM::setProperty` handles **only** `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` and
  `CAP_PROP_FPS`; everything else falls to `default: break` and returns false
  ([source: cap_avfoundation_mac.mm L505-552](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_avfoundation_mac.mm)).
  `getProperty` likewise returns 0 for anything except width, height, fps and format
  (L474-503). So `CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_EXPOSURE`, `CAP_PROP_AUTO_WB` and
  `CAP_PROP_FOURCC` do nothing on a Mac camera.
- The exposure/WB code that does exist (`CAP_PROP_IOS_DEVICE_EXPOSURE`,
  `CAP_PROP_IOS_DEVICE_WHITEBALANCE`) lives in `cap_avfoundation.mm`, the **iOS** backend
  ([source: cap_avfoundation.mm L500-600](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_avfoundation.mm)).
- Issue tracker agrees: "It seems that AVFoundation is not supporting exposure but only resolution
  and fps" ([issue #9738](https://github.com/opencv/opencv/issues/9738)).
- Workarounds: (a) a separate tool that sends UVC control requests over IOKit (e.g. the open-source
  `uvc-util`; **unverified with the C270**, and a setting may be reset when OpenCV opens the device);
  (b) do all calibration and the real run on Linux, where V4L2 exposes the controls. If the turret's
  computer will be Linux, (b) is the simple answer.
- The macOS backend opens the session with `AVCaptureSessionPresetMedium` and only switches size if
  you set width and height (L433, L446-470); it requests 32BGRA output and lets AVFoundation decode,
  so you cannot choose MJPG vs YUYV from OpenCV (no FOURCC case in the camera setter).

### Linux, V4L2 (`CAP_V4L2`)

- `CAP_PROP_AUTO_EXPOSURE` maps to `V4L2_CID_EXPOSURE_AUTO`, `CAP_PROP_EXPOSURE` to
  `V4L2_CID_EXPOSURE_ABSOLUTE`, `CAP_PROP_AUTO_WB` to `V4L2_CID_AUTO_WHITE_BALANCE`,
  `CAP_PROP_WB_TEMPERATURE` to `V4L2_CID_WHITE_BALANCE_TEMPERATURE`
  ([source: cap_v4l.cpp L1646-1730](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_v4l.cpp)).
- **The value quirk.** By default values pass through raw, so you use the kernel menu values:
  `1` = manual, `3` = aperture priority (auto). C270-class UVC cameras do not offer `0` (full auto)
  ([spec: V4L2 camera controls, `V4L2_CID_EXPOSURE_AUTO`](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)).
  `V4L2_CID_EXPOSURE_ABSOLUTE` is in 100 µs units (same page).
  The famous `0.25` (manual) / `0.75` (auto) values come from OpenCV's legacy *normalised* mode,
  where `AUTO_EXPOSURE` is mapped over `Range(0, 4)`, so 0.25 -> 1 and 0.75 -> 3
  ([source: cap_v4l.cpp L1761-1773](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_v4l.cpp)). That mode is **off** by
  default: `normalizePropRange = getConfigurationParameterBool("OPENCV_VIDEOIO_V4L_RANGE_NORMALIZED", false)`
  (L931), and `CAP_PROP_MODE` toggles it (L1927-1929). Note the code comment at L388-394 still
  says "Enabled by default (as OpenCV 3.x does)", which is stale; trust L931. Issue #9738 shows
  users hitting exactly this 0.25-vs-1 confusion ([issue #9738](https://github.com/opencv/opencv/issues/9738)).
- Force the backend: "To use V4L2 properties you should force backend via `CAP_V4L2` parameter"
  (maintainer, [issue #9738](https://github.com/opencv/opencv/issues/9738)).
- Recipe: `cap.set(CAP_PROP_AUTO_EXPOSURE, 1)`, then `cap.set(CAP_PROP_EXPOSURE, value)`,
  `cap.set(CAP_PROP_AUTO_WB, 0)`, `cap.set(CAP_PROP_WB_TEMPERATURE, kelvin)`. Verify with
  `v4l2-ctl -d /dev/video0 --all`. UVC cameras also have `exposure_auto_priority`, which lets the
  camera drop frame rate to lengthen exposure in dim rooms; OpenCV has no mapping for it (it is not
  in `capPropertyToV4L2`), so set `exposure_auto_priority=0` with `v4l2-ctl`.

### Windows, DirectShow (`CAP_DSHOW`)

- `CAP_PROP_AUTO_EXPOSURE`: `cvRound(value) == 1` means auto; **any other value means manual**
  ([source: cap_dshow.cpp L3520-3530](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_dshow.cpp)).
- `CAP_PROP_EXPOSURE` is passed to DirectShow `CameraControl_Exposure` unchanged (L3627), whose unit is
  log2 seconds, e.g. `-6` is about 1/64 s
  ([spec: Microsoft, CameraControlProperty](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolproperty)).
- `CAP_PROP_AUTO_WB` and `CAP_PROP_WB_TEMPERATURE` are implemented (L3431-3441, L3590-3614).
- `CAP_PROP_SETTINGS` opens the driver's own settings dialog (L3631), handy for checking.
- `CAP_PROP_FOURCC` can be set (L3383, L3478), so MJPG can be requested.

### Windows, Media Foundation (`CAP_MSMF`, the default on Windows)

- `CAP_PROP_AUTO_EXPOSURE`: **non-zero means auto, 0 means manual**
  (`value != 0 ? VideoProcAmp_Flags_Auto : VideoProcAmp_Flags_Manual`,
  [source: cap_msmf.cpp L2418-2419](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_msmf.cpp)). This is the **opposite**
  of V4L2, where `1` means manual. Code shared across platforms must branch on
  `cap.getBackendName()`.
- `CAP_PROP_AUTO_WB` is not implemented in MSMF (no case in get/set; reported in
  [issue #19621](https://github.com/opencv/opencv/issues/19621), which found `get` returned -1 and
  every `set` failed). Use `CAP_DSHOW` on Windows if white balance must be locked.

### Summary table

| Backend | Manual exposure | Auto exposure | Lock WB | Choose MJPG |
|---|---|---|---|---|
| AVFoundation (macOS) | not possible | n/a | not possible | not possible |
| V4L2 (Linux) | `AUTO_EXPOSURE=1` (or 0.25 in normalised mode) | `3` (or 0.75) | `AUTO_WB=0` | `FOURCC=MJPG` |
| DSHOW (Windows) | `AUTO_EXPOSURE` != 1 | `1` | `AUTO_WB=0` | `FOURCC=MJPG` |
| MSMF (Windows) | `AUTO_EXPOSURE=0` | non-zero | not implemented | `FOURCC` sets the output format; **unverified** that it switches the camera's native stream |

## 7. Recording correctness

**FPS overlay burned into saved frames.** Any pixel drawn before `out.write(frame)` is in the
dataset forever. Footage recorded for testing a detector should be the raw sensor frame; draw HUD
text on a copy (`disp = frame.copy()`) used only for `imshow`. If the FPS is wanted in the
recording, save it in a sidecar file instead.

**Writer fps hard-coded.** `VideoWriter(..., fps, ...)` sets the container's frame rate; it is
"Framerate of the created video stream" ([videoio.hpp](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/include/opencv2/videoio.hpp)),
not a capture rate. If the loop actually delivers 15 fps and the file says 30, playback runs 2x fast
and any speed estimate taken from the file is wrong by 2x. Fixes: measure the real rate over the
first second or two (or over the whole clip and remux), and write that; and record timestamps
regardless (next point), because the real rate is not constant.

**Per-frame timestamps missing.** A Kalman filter needs `dt` between measurements; with dropped or
uneven frames a constant `1/30` is wrong. Write a CSV next to the video with
`frame_index, t_host_ns, t_driver_ms`:
- `t_host_ns`: `time.monotonic_ns()` taken right after `read()`/`grab()` returns. Works everywhere.
- `t_driver_ms`: `cap.get(CAP_PROP_POS_MSEC)`. On V4L2 this is the kernel buffer timestamp
  ([source: cap_v4l.cpp L1849-1853](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_v4l.cpp), set from `buf.timestamp`
  at L996-997); on MSMF it is the sample time (cap_msmf.cpp L2185-2186). On a macOS **camera** it is
  0, since the camera `getProperty` has no `POS_MSEC` case (cap_avfoundation_mac.mm L474-503; the
  L993 case is the file reader). So host time is the portable choice.

**Threaded capture.**
- On V4L2, OpenCV queues 4 driver buffers by default (`DEFAULT_V4L_BUFFERS 4`,
  [cap_v4l.cpp L299](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_v4l.cpp)). If processing is slower than the camera,
  `read()` returns frames up to about 4 frames (130 ms) old, which matters a lot for aiming. Either
  `cap.set(CAP_PROP_BUFFERSIZE, 1)` or a capture thread that keeps only the newest `(frame, t)`.
- On macOS the backend already drops late frames (`alwaysDiscardsLateVideoFrames = YES`,
  [cap_avfoundation_mac.mm L429](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_avfoundation_mac.mm)), so latency is
  less of a problem, but `read()` still blocks the loop.
- Recommended shape: thread A grabs and timestamps; the main thread detects and displays; a thread
  B with a bounded queue does `VideoWriter.write` so mp4 encoding never stalls capture. Python's GIL
  is released inside OpenCV calls, so this works.

**Does the C270 really deliver 1280x720 at 30 fps?** Logitech rates it "Max Resolution: 720p/30fps,
fixed focus" ([Logitech C270 spec page](https://logitech.com/en-qa/shop/p/c270-hd-webcam)). That rate
needs MJPG. Uncompressed YUYV is 2 bytes/pixel: 1280 x 720 x 2 x 30 = 55.3 MB/s, while the largest
USB 2.0 high-speed isochronous endpoint carries 3 x 1024 bytes per 125 µs microframe = 24.6 MB/s
([spec: USB 2.0, section 5.6.4](https://www.usb.org/document-library/usb-20-specification)). So YUYV
720p tops out near 13 fps in theory; a user's `v4l2-ctl` listing for a C270 reportedly shows 7.5
and 5 fps for YUYV 1280x720 ([secondary: gentoo-user thread](https://archives.gentoo.org/gentoo-user/message/1f9ab2ced0cfcb62b9d7247d7abb0e74), **unverified**: page would not load, seen via search snippet only).
Run `v4l2-ctl --list-formats-ext` on the real camera to confirm.

Why this bites on Linux: OpenCV's V4L2 autosetup tries `YUYV` **before** `MJPG` in its format list
([cap_v4l.cpp L602-624](https://github.com/opencv/opencv/blob/4a401fd4aa7200166868da4a7e84566a04daa238/modules/videoio/src/cap_v4l.cpp)), so unless you ask for MJPG you get
YUYV and a low frame rate at 720p. Set it explicitly, before the size:

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
```

Also note the C270's exposure priority (section 6): in a dim room auto exposure lowers the frame rate
regardless of format. Locking exposure fixes that too.

On macOS you cannot choose the format; measure the delivered rate (count frames over 5 s with
nothing else in the loop) rather than trusting `CAP_PROP_FPS`, which only reports the
`activeVideoMaxFrameDuration` (cap_avfoundation_mac.mm L489-492).

---

## Kevin audit

### Task 1: "Research classical CV and how to use OpenCV to find balloons (colour thresholding, contours)"

**Verdict: partly correct.** The basics are right and the weaknesses list is honest
(hue range 0-179, low saturation makes hue unreliable, `RETR_EXTERNAL`, bounding boxes for
YOLO compatibility, and the largest-contour false-positive risk). But the research covers one
colour on a notebook, has no mask cleanup or shape filtering, and the code does the very thing the
notes warn against.

Gaps:

1. `docs/classic_CV_research.md:2` says hue is 0-179 but never mentions that **red wraps around** and
   needs two ranges; no multi-colour plan anywhere. `scripts/hsv_testing.py:21-22` has one hard-coded
   cyan range.
2. `docs/classic_CV_research.md:11-13`: ranges were read off a cyan notebook, not a balloon, under
   auto exposure; no calibration tool (trackbars) and no saved per-colour config.
3. `scripts/hsv_testing.py:36-37`: no blur and no morphological open/close before `findContours`.
   This is the direct cause of the "30+ contours" in `docs/classic_CV_research.md:25`.
4. `scripts/hsv_testing.py:42-43`: `max(contours, key=cv2.contourArea)` returns one blob and is
   exactly the false-positive case the notes describe at `docs/classic_CV_research.md:25`. No area,
   circularity, aspect, solidity or `minEnclosingCircle` filter; multiple balloons are impossible.
5. `docs/classic_CV_research.md:20` mentions using width/height to filter shape, but no code does it.
6. `docs/classic_CV_research.md:23` lists glare, similar backgrounds and skin tone but gives no
   mitigation (close for highlights, S/V minimums, shape filters, persistence, ROI).
7. No mention of locking exposure / white balance anywhere; thresholds will drift with auto exposure,
   and on macOS OpenCV cannot lock them at all (section 6).
8. `scripts/hsv_testing.py:17-18` reads the size but never sets 1280x720 (unlike `record_webcam.py:18-19`),
   so on macOS it runs at the `AVCaptureSessionPresetMedium` default size. Thresholds and any
   pixel-size reasoning are tuned at a different resolution from the one recorded and used.
9. No comparison with HoughCircles or adaptive approaches (section 5).
10. Minor: `scripts/hsv_testing.py:45` draws the rectangle on `frame` before `blended_mask` is made at
    line 47, so the "Blend" view includes the green box; `target_FPS` at line 27 is unused; the FPS at
    lines 50-55 measures loop rate including four `imshow` calls, not camera rate.

### Task 2: "Get recording working with your webcam, with an fps counter on screen"

**Verdict: partly correct.** It records 1280x720 mp4, reads back the real size for the writer
(`scripts/record_webcam.py:20-21`, good), checks both open calls, and shows an FPS counter. But the
saved footage is not trustworthy as test data for the detector or tracker.

Gaps:

1. `scripts/record_webcam.py:56-61`: the FPS text is drawn on `frame` **before** `out.write(frame)`, so it
   is burned into every saved frame (the comment at line 56 says this is intended). Draw HUD on a copy.
2. `scripts/record_webcam.py:26,32-33`: writer fps hard-coded to `target_FPS = 30` instead of the measured
   rate; if the loop runs slower, playback is sped up and speeds/timings from the file are wrong.
3. No per-frame timestamps anywhere in `scripts/record_webcam.py:42-65`; a Kalman tracker needs real `dt`.
   Add a CSV sidecar with `monotonic_ns` (and `CAP_PROP_POS_MSEC` where the backend supports it).
4. `scripts/record_webcam.py:7,18-19`: no backend and no `FOURCC=MJPG` requested. On Linux V4L2 this gets
   YUYV, which cannot do 720p30 over USB 2.0 (about 7.5 fps reported). On macOS the format cannot be chosen,
   so the delivered rate must be measured.
5. `scripts/record_webcam.py:42-68`: capture, mp4 encode, and `imshow` all run in one thread; an encode
   stall lowers capture rate, and on V4L2 a slow loop reads frames up to 4 buffers old. No capture thread,
   no writer thread, no `CAP_PROP_BUFFERSIZE`.
6. No exposure / white-balance lock, so recorded footage has drifting colour, which makes it a poor
   calibration set for HSV ranges.
7. Minor: `scripts/record_webcam.py:36-38` exits on writer failure without `cap.release()`; the FPS shown
   (lines 49-54) is loop rate including write and display, which is fine for a HUD but should be labelled.

### Recommended classical detector recipe

Preconditions: camera opened with an explicit backend, MJPG at 1280x720 where the backend allows it,
exposure and white balance locked (Linux V4L2 or Windows DSHOW; on macOS use an external UVC tool or
accept drift), and HSV ranges calibrated per balloon colour with a trackbar tool on footage recorded
under that locked setting.

```python
import cv2, numpy as np

# Calibrate these; values are starting points only.
CLASSES = {
    "red":    [((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (179, 255, 255))],
    "orange": [((10, 100, 60), (22, 255, 255))],
    "yellow": [((22, 100, 60), (35, 255, 255))],
    "green":  [((35, 80, 50), (85, 255, 255))],
    "blue":   [((85, 80, 50), (130, 255, 255))],
    "purple": [((130, 80, 50), (160, 255, 255))],
    "pink":   [((160, 80, 60), (170, 255, 255))],
}
K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
AREA_MIN, AREA_MAX = 600, 10_000   # balloon ~40-65 px wide at 720p

def detect(frame_bgr):
    hsv = cv2.cvtColor(cv2.medianBlur(frame_bgr, 5), cv2.COLOR_BGR2HSV)
    out = []
    for name, ranges in CLASSES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE)  # fills specular holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if not AREA_MIN <= a <= AREA_MAX:
                continue
            p = cv2.arcLength(c, True)
            circularity = 4 * np.pi * a / (p * p)
            x, y, w, h = cv2.boundingRect(c)
            solidity = a / max(cv2.contourArea(cv2.convexHull(c)), 1)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            fill = a / (np.pi * r * r)
            if circularity >= 0.6 and 0.6 <= w / h <= 1.3 and solidity >= 0.85 and fill >= 0.5:
                m = cv2.moments(c)
                out.append((name, m["m10"] / m["m00"], m["m01"] / m["m00"], r, a))
    return out  # every balloon, not just the largest
```

Around it: a capture thread that stamps each frame with `time.monotonic_ns()`, a tracker (Kalman per
detection, associated by nearest centre) that uses the real `dt` and requires a few frames of
persistence before a target is confirmed, and an optional HoughCircles check on the cropped
grayscale ROI (`minRadius` about 15, `maxRadius` about 40) when false positives from clutter remain.
Footage for tuning: raw frames (no HUD), measured fps in the writer, timestamps CSV alongside.
Known limit: white, black, silver, pastel and foil balloons will not be found by hue; that needs
Hough on gray, background subtraction, or the learned detector.
