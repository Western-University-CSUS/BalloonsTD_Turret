# YOLO26 for Pinpoint: fact check and model/export choice

> [!NOTE]
> **Status (2026-11-25):** the research behind the YOLO model decision. Where the team decided differently, **[ROADMAP.md](../ROADMAP.md) wins**:
> - **Hardware:** the turret computer is most likely a **Raspberry Pi**, not a laptop. Export for the Pi is **NCNN or ONNX**, picked by measuring. The laptop table in §6 applies only to dev machines (Mac: CoreML).
> - **NCNN caveat:** per §6, the exporter turns the NMS-free head **off** for NCNN, so an NCNN export runs with NMS. Check this when the Pi benchmark happens.
> - **Licence:** the AGPL-3.0 `LICENSE` is deferred to the end of the project (not added before the dependency, as §8 suggests).
> - **Model, imgsz, head:** adopted as written: YOLO26n, 640, validate `nms=True` vs `nms=False` on our own val.

Researched 2026-11-25 against Ultralytics `main` at commit `4a5ad7c` (package `ultralytics==8.4.218`, released 2026-11-23). Primary sources: the Ultralytics docs, GitHub releases, the source code, and the Ultralytics paper. Ultralytics changes fast (218 patch releases since YOLO26 launched), so pin the version in `pyproject.toml` and re-check anything here before relying on it.

Context: C270 at 1280x720 and 30 fps, balloons 40 to 65 px wide, several in view, 3 to 6 m, single class, laptop hardware unknown, 30 fps or better needed.

## 1. Name and release status

- The official name is **Ultralytics YOLO26**, written without a "v". It is not "YoloV26". Weights are `yolo26n.pt`, `yolo26s.pt` and so on ([model page](https://docs.ultralytics.com/models/yolo26/)).
- It is a **stable release**, not a preview. It shipped in `ultralytics 8.4.0` on **2026-01-14** ([v8.4.0 release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0)). The docs call it the "Newest SOTA Ultralytics models, released January 2026" ([model page](https://docs.ultralytics.com/models/yolo26/)).
- The official paper is *"Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models"* by Jocher et al., [arXiv:2606.03748](https://arxiv.org/abs/2606.03748), submitted 2026-06-02. [arXiv:2509.25164](https://arxiv.org/abs/2509.25164) (Sapkota et al.) is a third-party paper, not from Ultralytics.

## 2. NMS-free inference: default or not, and the real API

**NMS-free is NOT the default in current Ultralytics.** It was the default at launch, but the default changed later.

- At launch (v8.4.0) YOLO26 was billed as "native end-to-end NMS-free" ([v8.4.0](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0)), and the switch was a separate `end2end` argument.
- **v8.4.142 (2026-09-05)** merged `end2end` into `nms` ([release](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.142)):
  - `nms=None` **(default)** uses the one-to-many head, and Ultralytics runs NMS on its output.
  - `nms=True` uses the one-to-many head and builds NMS into supported exports.
  - `nms=False` uses the one-to-one NMS-free head.
  - `end2end=True` still works but is deprecated and maps to `nms=False`.
- Source confirmation: in `ultralytics/cfg/default.yaml`, `nms:` is unset, with the comment "None: external NMS (default); True: embed NMS on export; False: NMS-free head if available". `engine/predictor.py` passes `end2end=self.args.nms is False` to `AutoBackend`. In `engine/exporter.py`, `model.end2end = self.args.nms is False`.
- Both heads are always trained. The model YAML has `end2end: True` (`cfg/models/26/yolo26.yaml`), which only means the one-to-one head exists. Which head runs is chosen at predict or export time ([end-to-end guide](https://docs.ultralytics.com/guides/end2end-detection)).
- The NMS-free head returns `(N, 300, 6)`, meaning at most 300 detections ([model page](https://docs.ultralytics.com/models/yolo26/)).
- **Accuracy cost:** the NMS-free head scores 0.6 to 0.8 COCO mAP **lower** than the default head. For example, YOLO26n scores 40.9 with NMS and 40.1 without ([guide](https://docs.ultralytics.com/guides/end2end-detection)). The guide says the default head with NMS "favors accuracy".
- So Aadi's usage `nms=False` is the correct current argument. It gives lower latency and slightly *lower* accuracy, not higher.

## 3. Small-object changes (ProgLoss, STAL, DFL removal, MuSGD)

What each change is, from the [model page](https://docs.ultralytics.com/models/yolo26/), [paper](https://arxiv.org/abs/2606.03748) and source:

| Change | What it does | Does it matter for 20 to 65 px balloons? |
|---|---|---|
| **DFL removal** (`reg_max: 1`) | Makes the head simpler and easier to export. This is an export and speed change, not a small-object change. | No direct accuracy effect. It makes ONNX, CoreML and OpenVINO export cleaner. |
| **ProgLoss** (Progressive Loss) | Moves the loss weight from the one-to-many head to the one-to-one head as training goes on. In `utils/loss.py`, the one-to-many weight decays linearly from 0.8 to 0.1 across the epochs. | Not a small-object feature. It exists to train the NMS-free head. |
| **STAL** (small-target-aware label assignment) | "Guarantees positive coverage for small objects." In code (`utils/tal.py`), any ground-truth box whose side is smaller than `stride[1]` (16 px for P3 to P5) is enlarged to 16 px during label assignment. | **Barely.** Balloons are at least 20 px even after a 720p frame is shrunk to 640, so they are already above the 16 px floor. |
| **MuSGD** | A hybrid of Muon and SGD. | **Not used here.** With `optimizer=auto` (the default), `engine/trainer.py` picks MuSGD only when there are more than 10,000 training iterations. Otherwise it picks AdamW. A few-hundred-image balloon dataset gets AdamW. |

Neither the docs nor the paper abstract give a **small-object-specific number** such as APs for STAL or ProgLoss. The improvements they quote are overall COCO mAP and other tasks (segmentation +2.5 box AP, pose +7.2 AP, OBB +3.4 mAP over YOLO11) ([model page](https://docs.ultralytics.com/models/yolo26/)). For truly tiny targets, Ultralytics also ships a `yolo26-p2.yaml` variant with a stride-4 head. These balloons do not need it.

## 4. Size table (COCO, 640 px, from the [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/))

| Model | mAP 50-95 (NMS) | mAP 50-95 (e2e) | CPU ONNX ms | T4 TensorRT10 ms | Params M | FLOPs B |
|---|---|---|---|---|---|---|
| YOLO26n | 40.9 | 40.1 | 38.9 | 1.7 | 2.4 | 5.5 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 | 20.9 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 | 68.4 |

The CPU figures come from an Intel Xeon @ 2.00 GHz, which is the Colab or cloud CPU and is slower than a modern laptop CPU. Per the [guide](https://docs.ultralytics.com/guides/end2end-detection), the published NMS-free speeds are measured with `nms=False`.

## 5. YOLO26 vs YOLO11 ([YOLO11 docs](https://docs.ultralytics.com/models/yolo11/))

| Model | mAP | CPU ONNX ms | T4 TRT ms | Params M | FLOPs B |
|---|---|---|---|---|---|
| YOLO11n | 39.5 | 56.1 | 1.5 | 2.6 | 6.5 |
| YOLO26n | 40.9 (40.1 e2e) | **38.9** | 1.7 | 2.4 | 5.5 |
| YOLO11s | 47.0 | 90.0 | 2.5 | 9.4 | 21.6 |
| YOLO26s | 48.6 (47.8 e2e) | 87.2 | 2.5 | 9.5 | 20.9 |

What this means:
- **On CPU, YOLO26n is the real win.** It is about 31% faster than YOLO11n on CPU ONNX (38.9 ms vs 56.1 ms). Ultralytics' headline figure is "up to 43%" ([v8.4.0](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0)).
- At size s the CPU speeds are about the same.
- On an NVIDIA GPU, TensorRT speed is a wash. 11n is even 0.2 ms faster than 26n.
- The COCO mAP gain is about +1.4 to +1.6 with NMS and about +0.6 to +0.8 NMS-free. On a single-class balloon dataset, both models are likely to saturate, so the difference will be smaller.

## 6. Export formats and runtime per laptop

All four formats accept `nms` ([export docs](https://docs.ultralytics.com/modes/export/)). Per the [end-to-end guide](https://docs.ultralytics.com/guides/end2end-detection) and `exporter.py`, **ONNX, OpenVINO, CoreML and TensorRT all keep the NMS-free head with `nms=False`**. The exporter only turns end2end off for RKNN, NCNN, ExecuTorch, Paddle, IMX, EdgeTPU, QNN, NXP and STEdgeAI, for TensorRT < 8.5, and for LiteRT INT8.

| Laptop | Best runtime | Command | NMS-free works? |
|---|---|---|---|
| Intel CPU, or AMD CPU without an NVIDIA GPU | **OpenVINO** (Intel's own runtime, usually fastest on Intel). Plain ONNX Runtime is the fallback for AMD. | `model.export(format="openvino", nms=False)` | Yes |
| Apple Silicon | **CoreML** (uses the Neural Engine and GPU). ONNX Runtime on CPU is the fallback. | `model.export(format="coreml", nms=False)` (export on macOS) | Yes |
| NVIDIA laptop | **TensorRT FP16**. Build the engine on the target laptop, because engines are tied to the GPU and TensorRT version. | `model.export(format="engine", quantize=16, nms=False)` | Yes (TRT 8.5 or newer) |
| Unknown, or a first test | **ONNX**, which is portable | `model.export(format="onnx", nms=False)` | Yes |

The export docs don't rank the runtimes per chip. They only claim "up to 3x" CPU speedup for ONNX or OpenVINO and "up to 5x" GPU speedup for TensorRT ([export docs](https://docs.ultralytics.com/modes/export/)). **Benchmark on the real laptop** with `yolo benchmark model=best.pt imgsz=640`. `half=`/`int8=` are deprecated. Use `quantize=16` or `quantize=8` instead ([export docs](https://docs.ultralytics.com/modes/export/)).

## 7. Image size: 640 vs 960/1280 on 720p frames

- With `imgsz=640`, a 1280x720 frame is shrunk by a factor of 0.5. A 40 to 65 px balloon becomes **20 to 33 px**. That is right at COCO's "small" cut-off (32²), but it is still 2.5 to 4 cells on the stride-8 P3 grid, and a high-contrast, round, single-class target is easy to detect at that size. Exports are static and square by default (640x640, letterboxed), so 44% of the input is padding. Native `.pt` predict uses a rectangular input, about 640x384.
- With `imgsz=960`, balloons become 30 to 49 px. Compute grows by (960/640)² = **2.25x**. YOLO26n would take about 88 ms per frame on the benchmark Xeon, which **breaks 30 fps on CPU**. On an NVIDIA GPU it is still well under 33 ms.
- With `imgsz=1280` (native width), compute is **4x**. That makes sense only on an NVIDIA GPU.
- Compute scales with pixel count; this is arithmetic from the FLOPs above, not a measured benchmark.
- **Recommendation:** train and run at 640 first. Measure recall on the farthest (6 m) balloons in a validation split. Move to 960 only if far-balloon recall is poor *and* the laptop has the headroom. Train at the same `imgsz` you run at.

The 30 fps budget is 33 ms per frame for everything: camera read, preprocessing, inference, tracking and aiming. YOLO26n at 640 on the reference Xeon takes **38.9 ms**, which is already over budget. A modern laptop CPU with OpenVINO, or Apple Silicon with CoreML, will probably fit (an estimate, not yet measured). YOLO26s on CPU (87 ms) will not fit.

## 8. Licence (AGPL-3.0)

- Ultralytics code **and models trained with it** are AGPL-3.0 by default. The licence page says "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" ([license](https://www.ultralytics.com/license)).
- The free route covers "Academic research and university coursework" and "Personal projects", on the condition that you publicly release "the complete corresponding source code for the entire derivative work", including scripts, configs and, where applicable, weights ([license](https://www.ultralytics.com/license)).
- What this means for Pinpoint: a **public** club repo is fine if the repo is AGPL-3.0 compatible. Put an AGPL-3.0 `LICENSE` in the repo, or at least a compatible one, and keep the turret code that imports `ultralytics` public. Publishing `best.pt` is the safest reading. A club sponsor or company that wants closed code would need an Enterprise licence.
- The repo has no `LICENSE` file at the moment. Add AGPL-3.0 before adding `ultralytics` as a dependency.
- *(Not legal advice. This is the plain reading of the licence page.)*

## 9. Colab training

- The free tier GPU is not guaranteed. Google says the types available "vary over time" and that free notebooks run "at most 12 hours" with idle timeouts ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). In practice the free tier usually gets an **NVIDIA T4 (16 GB)**. The Ultralytics Colab page lists T4, L4 and A100 as runtime options ([Ultralytics Colab](https://docs.ultralytics.com/integrations/google-colab/)).
- Typical time is an **estimate, not from a source**. For about 300 to 1,000 single-class images, `yolo26n` at 640 for 100 epochs on a T4 takes roughly **20 to 60 minutes**. At 960 it takes about 2 to 2.5 times as long. Save checkpoints to Drive in case the session drops ([Ultralytics Colab](https://docs.ultralytics.com/integrations/google-colab/)).
- Snippet, based on the [model page](https://docs.ultralytics.com/models/yolo26/) example:

```python
!pip install "ultralytics==8.4.218"   # pin; behaviour of nms/end2end changed mid-2026
from google.colab import drive; drive.mount("/content/drive")
from ultralytics import YOLO

model = YOLO("yolo26n.pt")            # COCO-pretrained; fine-tune, don't train from scratch
model.train(
    data="/content/drive/MyDrive/pinpoint/balloons.yaml",  # nc: 1, names: ["balloon"]
    epochs=100, imgsz=640, batch=-1,   # batch=-1 = auto-fit GPU memory
    project="/content/drive/MyDrive/pinpoint/runs", save_period=10,
)
model.val(nms=False)                   # check the head you will actually deploy
model.export(format="onnx", nms=False) # portable artifact; re-export per laptop below
```

Resume after a disconnect with `YOLO(".../last.pt").train(resume=True)` ([Ultralytics Colab](https://docs.ultralytics.com/integrations/google-colab/)).

## 10. Is "NMS deletes overlapping balloons" a real risk here?

**The geometry.** Take two balloons of the same width `w`, side by side, with their centres a distance `d` apart. Their boxes have IoU = (w − d)/(w + d). NMS drops the lower-scored box only if its IoU with the kept box is greater than the threshold.

- At the **0.7** threshold (the validation default), a box is dropped only if d < 0.18·w. That means the centres are within **7 to 11 px** of each other, and the back balloon is more than 80% hidden.
- At the **0.5** threshold (the current *predict* default, `PREDICT_IOU = 0.5` in `utils/__init__.py`), a box is dropped only if d < w/3. That means the centres are within **13 to 22 px**, and the back balloon is about two-thirds hidden.

**Verdict: low risk, and mostly irrelevant.** Balloons that are next to each other or only touching have IoU well below 0.5 and survive NMS. Only a balloon that is mostly hidden behind another can be suppressed. The turret cannot hit that balloon until the front one is popped anyway, and the hidden balloon reappears on the next frame. The NMS-free head is not immune either. It has to *learn* one-box-per-object, and it scores lower on COCO, which includes crowded scenes ([guide](https://docs.ultralytics.com/guides/end2end-detection)). If overlapping balloons ever do cause trouble, raising `iou` to 0.7 is a one-argument fix.

**Reasons that matter more for the choice:**
1. **CPU latency on an unknown laptop.** YOLO26n is 31% faster than YOLO11n on CPU ONNX. This is the strongest reason to pick YOLO26.
2. **A clean export.** Without DFL and NMS, the exported model emits final boxes directly. ONNX, OpenVINO and CoreML need no NMS code on our side, so the runtime code is simpler.
3. **Steady latency.** With NMS-free output, post-processing time does not grow with the number of detections. This matters a little when many balloons are in view. NMS on a handful of boxes costs well under 1 ms, so it is not a big saving.
4. **Data quality and imgsz** affect far-balloon recall much more than the choice of head does.

## Aadi model-choice audit

| # | Claim | Verdict | Source and notes |
|---|---|---|---|
| 1 | "We will use YoloV26" | **Partly true** | The model exists and is stable (8.4.0, 2026-01-14), but its name is **YOLO26**, without the "v" ([v8.4.0](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0)). |
| 2 | "NMS can delete a real balloon because its box overlaps a neighbour's" | **Partly true** | It can happen in principle, but only when IoU is above 0.5 (predict default) or 0.7. That needs a balloon about 67 to 80% hidden, which the turret cannot target anyway (§10). |
| 3 | "YOLO26's NMS-free mode (nms=False) avoids that step entirely" | **Verified** | `nms=False` selects the one-to-one head with no NMS step (`default.yaml`, `predictor.py`, [v8.4.142](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.142)). It is **not the default**: `nms=None` runs NMS. |
| 4 | "YOLO26 adds training changes aimed at small objects" | **Partly true** | STAL targets small objects ([paper](https://arxiv.org/abs/2606.03748)), but it only enlarges boxes under 16 px, and our balloons are 20 px or more. ProgLoss and DFL removal are not small-object features. MuSGD is skipped for small datasets (it needs more than 10k iterations). No numbers for small objects are published. |
| 5 | "It is the same workflow… same few lines of code" | **Verified** | It uses the same `YOLO(...).train/val/predict/export` API ([model page](https://docs.ultralytics.com/models/yolo26/)). |
| 6 | "NMS is optional, so there should be more accuracy when detecting more balloons" | **Wrong** | The NMS-free head scores 0.6 to 0.8 mAP *lower* on COCO. The NMS head "favors accuracy" ([guide](https://docs.ultralytics.com/guides/end2end-detection)). Dropping NMS is a speed and simplicity choice, not an accuracy gain. |
| 7 | "YOLO26 natively supports end-to-end inference without NMS (nms=False)" | **Verified** | It is native and both heads are trained. The argument is `nms=False`; the older `end2end=True` is deprecated ([v8.4.142](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.142)). |
| 8 | "…which reduces latency when tracking balloons in real-time" | **Partly true** | It removes the NMS step and keeps latency steady, but with only a few balloons NMS costs under about 1 ms. The large CPU speedup over YOLO11 comes from the architecture (DFL removal, a lighter head), not mainly from skipping NMS. |

### Recommendation

- **Model: YOLO26n**, fine-tuned from `yolo26n.pt` on a single `balloon` class. The reason to pick it is CPU speed on an unknown laptop, not NMS. Try 26s only if the laptop has an NVIDIA GPU *and* n misses far balloons.
- **imgsz: 640** for training and inference. Move to 960 only if recall at 6 m is poor and the laptop measures under about 25 ms per frame at 960. That is realistic only with TensorRT.
- **Head:** compare `model.val()` with `model.val(nms=False)` on our own validation split. Deploy `nms=False` unless it loses noticeable recall. It makes exports simpler and needs no NMS code.
- **Export, chosen on the actual laptop:**
  - NVIDIA: `format="engine", quantize=16, nms=False`
  - Apple Silicon: `format="coreml", nms=False`
  - Intel CPU: `format="openvino", nms=False`
  - Anything else: `format="onnx", nms=False`
  - Then run `yolo benchmark` there. If the CPU path cannot reach 30 fps at 640, fall back to `imgsz=512`, or run detection on every other frame and track in between.
- **Housekeeping:** pin `ultralytics` (the meaning of `nms` changed in 8.4.142), and add an AGPL-3.0 `LICENSE` to the public repo before the dependency lands.
