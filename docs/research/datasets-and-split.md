# Balloon datasets and train/val/test split for the Pinpoint YOLO detector

> [!NOTE]
> **Status (2026-11-25):** the research behind the dataset and split decisions. Where the team decided differently, **[ROADMAP.md](../ROADMAP.md) wins**:
> - **imgsz:** we train and run at **640**, not ≥1280 as §4 suggests (that suggestion was reasoning, not a sourced fact, and 1280 is too slow for a Pi).
> - **Split:** by **session count**: 12 sessions → 7 train / 2 val / 3 test (8 sessions → 5 / 1 / 2). Test is demo-room footage, frozen and versioned (`test_v1` = placeholder room, `test_v2` = real demo room). No separate "reserve" session.
> - **Session id:** `YYYYMMDD_<room>_<n>`.
> - **Two stages:** Stage 1 trains on public data only (validated on Open Images' val, no test). Stage 2 adds our own sessions.
> - **Labelling:** CVAT, class `balloon` plus a `colour` attribute on every box.
> - **Dropped sources:** tm1901, rkuo2000, LVIS, Roboflow Universe.

Researched 2026-11-25. Target domain: indoor helium balloons, 3–6 m away, Logitech C270 at 1280x720, balloons about 40–65 px wide (about 3–5 % of frame width), colours unknown. Single class `balloon`. Ultralytics YOLO fine-tuned on Colab.

**Bottom line**

- No public dataset matches our domain. They are mostly close-up party, stock and Flickr photos. Use them **only in `train`**, as a generic "what is a balloon" prior.
- The single best public add-on is **Open Images V7 `Balloon`**: licensed, human-drawn boxes, and many small and grouped balloons. Second is the Kaggle **serhiibiruk** set: already YOLO txt, but its licence and provenance are murky and it contains Roboflow augmentation copies.
- The two other Kaggle links (tm1901, rkuo2000) are **byte-identical copies of the Matterport Mask R-CNN balloon set** (61 train + 13 val images). They add nothing beyond serhiibiruk, which already contains all 74 images.
- `val` and `test` must be **our own C270 footage**, split **by recording session**, never by frame. `test` should be held-out sessions, ideally recorded in the demo room, and frozen.

---

## 1. Dataset-by-dataset evaluation

How the numbers were obtained: the Kaggle web pages render client-side, so WebFetch returned nothing. I used the public Kaggle API JSON endpoints instead (`/api/v1/datasets/view/<ref>` and `/api/v1/datasets/list/<ref>`, paginated, no auth). They give the licence, total bytes and the full file list. Counts below come from those listings. Box counts for the Kaggle sets that need the actual label files are marked **estimate**.

### 1a. Kaggle `serhiibiruk/balloon-object-detection`

Source: https://www.kaggle.com/datasets/serhiibiruk/balloon-object-detection (API metadata retrieved 2026-11-25)

| Field | Finding |
|---|---|
| Images | **2,365** files: `Balloon/train/images` 2,256 and `Balloon/valid/images` 109. There is no test split. Uploaded 2024-05-13, 157 MB. |
| Unique source images | **588**. Filenames are Roboflow exports (`<orig>_jpg.rf.<hash>.jpg`), and most source images appear about 3x in train. That means **Roboflow augmentation copies are baked in**. |
| Boxes | **Estimate: about 10–15k.** Label .txt total is 587,598 B (train) and 15,206 B (valid). The median label file is 79 B, which is about 2 boxes. The largest is 23.8 kB, a crowd of balloons or polygon rows. I did not download the labels, so this is unverified. |
| Format | **Already YOLO txt** (`images/` + `labels/` + `data.yaml`). The Kaggle description says "COCO format", but the file list contradicts it: there are no JSON files. Check that `data.yaml` has `nc: 1`. If some rows are polygons (more than 5 numbers), Ultralytics detect will still convert segments to boxes, but check this first. |
| Licence | Kaggle field says **MIT**. The description says images are "sourced from various publicly available datasets". Filenames include Flickr IDs (the Matterport set), stock-shop product names (`10x-luftballons-in-schwarz-metallic`, `12-sade-balon-10lu-kirmizi`) and Turkish blog images. The **MIT label is not trustworthy for the images**. That is fine for internal club training. Do not redistribute it. |
| Style | Mixed web images: party bunches, product shots, outdoor festivals. Mostly large, close balloons. |
| Domain match | **Low to medium.** Colour and shape variety is good. Scale and viewpoint are wrong: few balloons are 3–5 % of the frame width, and there is no C270 noise or blur. |
| Leakage warning | **9 of the 82 valid source images also appear in train** (as augmented copies). Its own `valid` split is therefore leaky. **Do not use its `valid` folder as our validation set.** Put all of it in train. |
| Overlap | **74 source images have Flickr-style names**, which is exactly the Matterport 61 + 13. It already contains the whole Matterport set. |

### 1b. Kaggle `tm1901/balloon`

Source: https://www.kaggle.com/datasets/tm1901/balloon (API metadata)

| Field | Finding |
|---|---|
| Images | **74**: `balloon/train` 61 and `balloon/val` 13. Uploaded 2020-10-17, 39.0 MB. |
| Boxes | **305 polygons**: 255 train + 50 val. Verified by downloading the canonical Matterport `balloon_dataset.zip` and counting VIA regions. |
| Format | **VIA (VGG Image Annotator) JSON polygons**: one `via_region_data.json` per split (97,007 B train, 19,756 B val). |
| Licence | Kaggle field: **"Unknown"**. Upstream is the Matterport repo (MIT code). The images are Flickr photos with no per-image licence shipped. |
| Style | High-resolution Flickr photos, mostly large, close, outdoor or party balloons. Image widths are 768–2048 px (mostly 1024 or 2048). Median balloon box width is 169 px (train) and 141 px (val), roughly 10–15 % of the frame width, about 3x our relative size. |
| Domain match | **Low.** It is also tiny. |
| Verdict | **Identical to rkuo2000/balloon-dataset**: same total bytes (39,001,256) and the same per-file sizes. Both are re-uploads of the Matterport `balloon_dataset.zip`. It is also already inside serhiibiruk. **Skip, or use only one copy.** |

### 1c. Kaggle `rkuo2000/balloon-dataset`

Source: https://www.kaggle.com/datasets/rkuo2000/balloon-dataset (API metadata)

Confirmed: **this is the Matterport Mask R-CNN balloon dataset.** It has the same 61 train + 13 val images, the same two `via_region_data.json` files and the same 39,001,256 bytes as tm1901. Uploaded 2020-10-04, and its Kaggle licence is "Unknown". The canonical source is the Matterport release asset `balloon_dataset.zip` (https://github.com/matterport/Mask_RCNN/releases/tag/v2.1, "The Balloon Color Splash sample, along with dataset and trained weights"). I downloaded it to verify the counts above. Aadi's link text says "serhiibiruk" but the URL points here. That is a listing mistake, and it matters: serhiibiruk is roughly 8x larger in unique images, about 32x larger with augmentation copies, and is already in YOLO format.

**VIA polygon to YOLO txt conversion** (Matterport format: `regions[*].shape_attributes.all_points_x/y`). For each region:

```python
x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
line = f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}"
```

`W` and `H` come from the image, not from the JSON. Alternatively, upload the images and VIA JSON to Roboflow and export as "YOLOv8/YOLO11". Roboflow imports VIA.

### 1d. Open Images V7, class `Balloon` (MID `/m/01j51`)

Sources: class list https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv. Boxes: `oidv6-train-annotations-bbox.csv`, `validation-annotations-bbox.csv` and `test-annotations-bbox.csv` under https://storage.googleapis.com/openimages/. Licence: https://storage.googleapis.com/openimages/web/factsfigures_v7.html. I counted rows myself by streaming the official CSVs.

| Split | Images | Boxes | `IsGroupOf=1` | Box width < 5 % of image |
|---|---|---|---|---|
| train | 3,262 | 13,505 | 1,450 | 4,228 |
| validation | 51 | 137 | 17 | 20 |
| test | 121 | 485 | 44 | 156 |

- **Licence:** "The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license", with the caveat "you should verify the license for each image yourself" (factsfigures_v7). This is the cleanest licence of anything here.
- **Format:** CSV with normalised `XMin, XMax, YMin, YMax`. FiftyOne and the OIDv4 toolkit convert it (see below).
- **Style:** Flickr photos. There are many party and street scenes with **small and distant balloons**. Across train, 4,228 of 13,505 boxes (31 %) are narrower than 5 % of the image width, which is our regime. Also note `Hot-air balloon` is a **separate class** (it is a separate class in the Ultralytics `open-images-v7.yaml`), so it will not pollute ours.
- **Caveats:** `IsGroupOf=1` boxes cover a whole bunch with one box. They teach the wrong thing for a turret that must aim at a single balloon. **Drop group-of boxes**, or drop the images that contain them. Open Images labels are also not exhaustive. Expect some unlabelled balloons, which become false "background" signal.
- **Domain match:** **Medium.** It is the best public source for small balloons, but it is still outdoor and party scenes, not a C270 indoors.

**Download only this class: FiftyOne (recommended, handles V7)**
(https://docs.voxel51.com/integrations/open_images.html, https://docs.voxel51.com/user_guide/export_datasets.html)

```python
import fiftyone as fo, fiftyone.zoo as foz
ds = foz.load_zoo_dataset("open-images-v7", split="train",
        label_types=["detections"], classes=["Balloon"], max_samples=2000)
# drop group-of boxes; also drops other classes so only Balloon is exported
from fiftyone import ViewField as F
view = ds.filter_labels("ground_truth",
        (F("label") == "Balloon") & (F("IsGroupOf") == False))
view.export(export_dir="oi_balloon", dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth", split="train", classes=["Balloon"])
```

`classes` "loads only samples containing at least one instance of a listed class". `max_samples` caps each split. The export call mirrors the documented `YOLOv5Dataset` example. Check the label field name with `ds.get_field_schema()`. Note: an Ultralytics docs commenter reported that class filtering still downloaded the default classes (https://docs.ultralytics.com/datasets/detect/open-images-v7/). That is unverified, so check the result. Do not run the Ultralytics `open-images-v7.yaml` auto-download: it pulls the full set, about 561 GB.

**OIDv4 toolkit (alternative)** (https://github.com/EscVM/OIDv4_ToolKit): `python3 main.py downloader --classes Balloon --type_csv train --limit 2000`. Its README documents **only Open Images v4**, and labels come out as `name left top right bottom` in **pixel** coordinates. Converting to YOLO means `cx=(l+r)/2/W` and so on. FiftyOne is simpler.

### 1e. COCO / LVIS / Objects365

- **COCO (80 classes): no balloon class.** `coco.yaml` in Ultralytics has no "balloon" (https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml). A stock COCO-pretrained YOLO therefore has never been trained to say "balloon". Fine-tuning is required.
- **LVIS v1: `balloon` is category id 43, frequency "f" (frequent), in 210 train images** (detectron2 `lvis_v1_categories.py` and `lvis_v1_category_image_count.py`). LVIS reuses COCO images and is federated (not exhaustively labelled per image). It is usable, but small and fiddly. Low priority.
- **Objects365: has a `Balloon` class (index 90)** in the Ultralytics `Objects365.yaml`. Objects365 is huge and its licence is research-oriented. Not worth the download for one class. Mentioned for completeness only.

### 1f. Roboflow Universe

Universe pages returned **HTTP 403** to WebFetch, so everything here comes from search snippets and is **unverified**. Candidates include:

- `balloon-kc3lz/balloon-o8xng`: about 697 images, 4 classes
- `astekdatasetproject/balloon-detection-pse6c`: CC BY 4.0
- `balloon/balloons-3yqzy`: CC BY 4.0
- `balloon-emler/happy-balloon`: Google Images and Unsplash, hand-labelled
- `tramola/real-world-balloon`
- `detection-the-red-balloon-7ikea/balloon-detection-p6e6u`
- `oktober13/balloons-v8nvr`
- several tiny ones (9 and 27 images)

Universe sets export directly to YOLO txt (format "YOLOv8"/"YOLO11"). Most are small, single-author web scrapes that overlap the same Google and Flickr pool. Before using one, open the page and check four things: the licence, the image count *before* augmentation, whether the frames are indoor or small-object, and a de-dupe against serhiibiruk. Some Universe sets are **CTF/air-defence "balloon"** projects (for example `hava-savunma` means "air defence"). They are outdoor sky shots and probably a poor match.

### Summary table

| Source | Unique imgs | Boxes | Format → YOLO | Licence | Domain fit | Use |
|---|---|---|---|---|---|---|
| serhiibiruk (Kaggle) | 588 (2,365 with aug copies) | about 10–15k (est.) | already YOLO | "MIT" (dubious) | low–med | train only, de-dupe aug copies |
| tm1901 (Kaggle) | 74 | 305 (verified) | VIA polygons → min/max | Unknown | low | skip (dup) |
| rkuo2000 (Kaggle) | 74 | same | same | Unknown | low | skip (dup) |
| Open Images V7 Balloon | 3,434 | 14,127 | CSV → FiftyOne export | CC BY 4.0 / CC BY 2.0 | **med (best public)** | train only, drop group-of |
| LVIS balloon | 210 | ? | COCO JSON → `convert_coco` | CC BY 4.0 annots, COCO imgs | low–med | optional |
| Roboflow Universe | varies | varies | native YOLO export | varies (unverified) | varies | vet one by one |

---

## 2. Split practice

### 2a. Three splits, three jobs

- **train**: the weights are fitted on it.
- **val**: used *during development* to choose epochs, hyperparameters, confidence threshold and which data to add. Ultralytics evaluates on it every epoch and picks `best.pt` from it.
- **test**: touched **only to report final numbers**. Google's ML Crash Course: the validation set "performs the initial testing on the model as it is being trained", and the test set is a final double-check. It also warns that "test sets and validation sets still 'wear out' with repeated use" (https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets).

Without a test split, the number you report is the val number you tuned on, so it is optimistically biased.

### 2b. Typical ratios

Ultralytics: "A common split is 70% for training, 20% for validation, and 10% for testing", described as a starting point (https://docs.ultralytics.com/guides/preprocessing_annotated_data/). With small datasets, the val and test sets need a **minimum absolute size** (a few hundred boxes) to give stable mAP, more than they need a percentage. Size them by count, not by ratio.

### 2c. Video-frame leakage: split by session, sample sparsely

Consecutive webcam frames are near-duplicates. If frames from one clip land in both train and val, val mAP measures memorisation.

- Ultralytics training tips: "keep near-duplicate groups entirely within one split" (https://docs.ultralytics.com/guides/model-training-tips/). Also: "Ensure that validation and test images never appear in the training set to avoid overly optimistic metrics" (https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/).
- Ultralytics (Glenn Jocher, data-collection guide thread): "split by original scene before tiling to prevent leakage" (https://docs.ultralytics.com/guides/data-collection-and-annotation/).
- scikit-learn, grouped data: "all the samples in the validation fold come from groups that are not represented at all in the paired training fold" (`GroupKFold` / `GroupShuffleSplit`, https://scikit-learn.org/stable/modules/cross_validation.html). Here, the group is **the recording session**: same room, lighting, balloon set and camera placement.
- Google MLCC: "delete any examples in the validation set or test set that are duplicates of examples in the training set."
- Ultralytics preprocessing: "Split the dataset before applying any augmentation." This is exactly the bug in the serhiibiruk `valid` folder.

**Practice:**

1. Give each recording a session ID (`YYYYMMDD_room_lighting_take`).
2. Assign **whole sessions** to train, val or test.
3. Extract frames sparsely: about 1–2 fps, or only when the scene changes. Then drop near-duplicates, for example with a perceptual-hash or SSIM threshold, or FiftyOne's uniqueness/similarity tools.

Ten distinct frames per session beat 300 nearly identical ones.

### 2d. Role of external (public) data

Put it in **train only**. Andrew Ng, *Machine Learning Yearning*: "Choose dev and test sets to reflect data you expect to get in the future and want to do well on". Training data "does not have to be drawn from the same distribution as what we really care about (the dev/test set distribution)" (deeplearning.ai PDF: https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf). Ultralytics: data "must be representative of deployed environment" (tips_for_best_training_results).

Putting web images into val would make `best.pt` selection reward performance on party photos rather than on our C270 frames. Optionally, report a **separate** "public-val" number as a sanity check, but never select on it.

### 2e. The test set: held-out target-domain footage, frozen

- Record it **in the demo room if at all possible**, or at least a room with similar lighting and distance. Use the C270 at 720p, with balloons at 3–6 m and the colours we expect. Include hard cases: backlit, partial occlusion, two balloons close together, a moving turret (motion blur), and balloon-free frames (false-positive check).
- **Freeze it.** Version it (a checksum list of files), never train or tune on it, and evaluate every detector on exactly the same frames. The classical HSV/contour pipeline and each YOLO run must use the same frames, metric and IoU threshold, so the comparison is fair. Classical CV has no "training", but its thresholds are tuned. Tune them on **val** too, not on test.
- Ultralytics can score it directly: `model.val(data="balloon.yaml", split="test")`. The YAML `test:` key is supported, and `train`/`val`/`test` can each be "a directory, an image file, a list of directories or image files" (https://docs.ultralytics.com/datasets/detect/).
- Because test sets "wear out" (Google MLCC), keep one **demo-room session in reserve** and record it as late as possible, ideally in the days before March 2027, as the final go/no-go check.

### 2f. How many own-footage labels?

Ultralytics guidance comes in tiers:

- "a few hundred annotated objects per class is enough to start experimenting with transfer learning" (data-collection-and-annotation FAQ)
- About 50–200 images per class "for a usable starting signal", and 500–2000+ for robustness (Glenn Jocher comment on model-training-tips, 2026-01-18)
- The production-grade bar is "≥ 1500 images per class" and "≥ 10000 instances" (tips_for_best_training_results)
- "about 0-10% background images to help reduce FPs"
- "All instances … must be labeled. Partial labeling will not work."

For one visually simple class with a COCO-pretrained backbone, the plan below is about **300–600 labelled own frames** in train. Public data makes up the bulk of the instance count, and own data supplies the domain match. Grow it from val failures ("train on a small pilot subset, check validation failures, then add data specifically where the model breaks"). This is a planning estimate, not a guarantee.

---

## 3. Aadi dataset/split audit

| Item | Aadi's version | Problem | Fix |
|---|---|---|---|
| Split | "Most like 80% Test / 20% Val" | There is **no train split** (80 % "test" was almost certainly meant as train). There is **no true test split**, so final numbers would be tuned-on val numbers. | Use **train / val / test**. See the plan below. |
| Split unit | Percent of images, implied random | A random split of video frames leaks near-duplicates across splits (2c). | Split **by recording session**, then sample frames sparsely and de-dupe. |
| Public data placement | Not stated | Mixing web images into val or test makes model selection and reporting measure the wrong domain (2d). | Public data goes in **train only**. |
| Dataset link | Link text "serhiibiruk", URL `rkuo2000/balloon-dataset` | The URL is the 74-image Matterport set (VIA polygons, licence Unknown), not the 588-unique/2,365-file serhiibiruk set. tm1901 is a second copy of the same thing. | Use serhiibiruk (or Open Images). Drop rkuo2000/tm1901 as duplicates. |
| serhiibiruk `valid` | (implicitly reused) | It contains Roboflow augmentation copies, and 9 of its 82 valid sources also appear in train. | Pool everything into train. De-dupe on the `<orig>` filename stem to about 588 originals, and let Ultralytics augment online. |
| Licence | Not checked | serhiibiruk is "MIT" but the images are scraped. Matterport copies are "Unknown". | Fine for internal training. Use Open Images (CC BY) for anything we publish. Do not redistribute the Kaggle images. |

---

## 4. Recommended dataset plan and split

**Data sources**

1. **Own C270 footage (core).** At least 8–12 recording sessions across rooms, lighting, distances of 3–6 m, at least 4 balloon colours, and a static or moving turret. Extract at 1–2 fps, de-dupe, and label every balloon with tight boxes. Include about 5–10 % balloon-free frames.
2. **Open Images V7 `Balloon`, train split.** Download via FiftyOne, drop `IsGroupOf` boxes, and optionally keep only images where some box is less than 10 % of the image width. That leaves roughly 2,273 train images with at least one non-group box. 1,070 of them have a non-group box narrower than 10 % of the image, counted from the CSV.
3. **serhiibiruk (optional).** De-dupe to about 588 originals by filename stem. Do not use its `valid/`. Exclude the 74 Flickr-ID Matterport images if you also pull Matterport (simplest is not to pull Matterport at all).
4. Skip tm1901, rkuo2000, Objects365 and LVIS unless you need more.

**Split (assign whole sessions, never frames)**

| Split | Contents | Size target |
|---|---|---|
| **train** | All public data, plus about 70 % of own *sessions* | own: about 300–600 frames, public: about 1–2k images |
| **val** | About 20 % of own sessions (at least 2 sessions, different from train). Own footage only. | about 100–200 frames, at least 300 boxes if possible |
| **test** | About 10 % of own sessions (at least 1–2 sessions), **recorded in or like the demo room**. Own footage only. Frozen and checksummed. Shared benchmark for classical vs YOLO. | about 100–200 frames |
| reserve | One late demo-room session, never looked at until the final check | about 50–100 frames |

Example `balloon.yaml`:

```yaml
path: datasets/balloon
train: [own/train/images, public/oi_balloon/images, public/serhiibiruk/images]
val: own/val/images
test: own/test/images
names: {0: balloon}
```

Train with `imgsz` of at least 1280 (or train at 1280 and infer at 1280). Balloons are 40–65 px in a 1280-wide frame, so at the default `imgsz=640` they shrink to about 20–32 px. That is a hypothesis to check on val, not a cited fact. Keep the whole pipeline (classical and YOLO) scored on `split="test"` with the same frames every time.

---

### Unverified or open items

- serhiibiruk box count (estimated from label byte sizes) and whether any rows are polygons. Download and run `wc -l` on the labels to confirm.
- Roboflow Universe image counts and licences (403 to the fetcher).
- The Ultralytics commenter report that FiftyOne's `classes=` filter still downloaded the default classes.
- The `imgsz` recommendation is reasoning, not a sourced claim.
