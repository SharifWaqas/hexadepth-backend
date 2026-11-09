# Model card: HexaDepth detectors on real side-scan sonar

HexaDepth turns a side-scan sonar image into one 64-byte packet per detected object. The website and `scripts/evaluate_real.py` can use four interchangeable detectors (`hexadepth/detectors.py`). This card describes them and the measured results.

## Main detector: `authors_yolov4`

| | |
|---|---|
| Model | YOLOv4 (Darknet), `yolov4-custom.txt` (cfg) + `yolov4-custom_5000.weights` (5,000 iterations) |
| Trained by | Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Shipped with the dataset. |
| Licence | CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ |
| Source | *Side-scan sonar imaging data of underwater vehicles for mine detection* [dataset], https://doi.org/10.6084/m9.figshare.24574879. Paper: Data in Brief, 53, 110132, https://doi.org/10.1016/j.dib.2024.110132 |
| Classes | 0 MILCO (mine-like object), 1 NOMBO (non-mine bottom object) |
| Input | 512 × 512 (from the cfg), plain resize, RGB, 0–1 |
| Runtime here | OpenCV 4.14 DNN (Darknet importer), CPU only. OpenCV 5.0 removed this importer, so `requirements.txt` pins `<5`. |
| Thresholds | confidence 0.25 (Darknet's default detection threshold), class-wise NMS IoU 0.45. **Not chosen on test images.** |
| Changes by HexaDepth | None to the weights or cfg. We added outline extraction inside each box and 64-byte packing. |
| Packet class ids | 10 = MILCO, 11 = NOMBO (ids 0–4 remain HexaDepth's rule-based shape classes; `packet.py` unchanged) |

**Training data: unknown.** The authors' notebook (`Real_time_object_classifier.ipynb`) builds `train.txt` / `test.txt` from zip files on their Google Drive. Those lists are not in the dataset, so no held-out set can be built for this model. Its validation run in the notebook reports 252 images, 174 objects and mAP@0.5 = 75.47%, but does not name the images. **Every number for this detector may include images the model was trained on.**

## Other detectors

- **`yolo`:** YOLOv8n fine-tuned by HexaDepth on Google Colab (`notebooks/train_yolo_colab.ipynb`). It was trained on the Batch D2 train split: 2010 + 2015 + 2017, minus a stratified 15% validation slice (seed 0). Settings: imgsz 1024, 80 epochs, patience 20, seed 0. It has **never seen 2018 or 2021**. File `models/yolo_milco_v8n.pt` (not in git). Runs on CPU with Ultralytics' default predict settings (confidence 0.25, NMS IoU 0.7).
- **`classical`:** the OpenCV pipeline (`hexadepth/vision.py`, profile `real`), tuned on 2015 only. It reports at most one object per image.
- **`oracle`:** not a detector. It uses the labelled boxes as detections to show what the 64-byte packet preserves.

## How each detection becomes 64 bytes

1. The detector produces boxes.
2. `vision.outline_in_box` extracts the object's outline inside each box: highlight (brighter than 1.4 × the local seabed) plus acoustic shadow (darker than 0.8 ×). It then cleans the mask with morphology, takes the largest contour and simplifies it to at most 20 vertices.
3. The outline is stretched so its bounding box equals the detector box. The shape adds detail but never moves or resizes the detection. If no usable contour is found, the box rectangle is sent instead.
4. The packet holds the outline on the 0–255 grid, the centre offset, the extent, the class and the confidence, with a CRC-16. It is decoded from the bytes only.

## Evaluation (measured; `results/real_eval.md`)

Data: MILCO side-scan sonar, survey years **2018 + 2021**: 612 images, 139 with objects, 473 without, 191 objects (145 MILCO, 46 NOMBO). All detectors were run once on the same images with identical rules: decoded outline's bounding box vs labelled box, greedy one-to-one matching, every image counted. CPU: Intel Core i7-8565U (8 threads), single process. Run on 2026-11-07.

| Detector | Trained on these images? | Recall @0.5 | Precision @0.5 | Recall @0.3 | False alarms / image @0.5 | MILCO recall @0.5 | NOMBO recall @0.5 | CPU median / p95 |
|---|---|---|---|---|---|---|---|---|
| Authors' YOLOv4 | **Unknown, may include** | 91.6% (175/191) | 91.6% | 94.8% (181/191) | 0.026 | 91.0% (132/145) | 93.5% (43/46) | 799 / 1,060 ms |
| Our YOLOv8n | No | 18.8% (36/191) | 29.3% | 22.0% (42/191) | 0.142 | 18.6% (27/145) | 19.6% (9/46) | 238 / 348 ms |
| Classical (`real`) | No | 0.5% (1/191) | 0.2% | 5.2% (10/191) | 0.775 | 0.0% (0/145) | 2.2% (1/46) | 55 / 438 ms |
| Oracle (labelled boxes) | n/a | 99.0% (189/191) | 99.0% | 99.0% | 0.003 | 98.6% | 100.0% | n/a |

- **Class agreement** among matched objects at IoU 0.5: authors' YOLOv4 100.0%, our YOLOv8n 83.3%.
- **False alarms:**
  - Authors' YOLOv4: 16 at IoU 0.5, of which 1 on images with no object.
  - YOLOv8n: 87, of which 49 on images with no object.
  - Classical: 474, of which 364 on images with no object.
- **Packets:** every packet was exactly 64 bytes and decoded with a valid CRC.
- **Size per matched object** (authors' YOLOv4, 181 objects at IoU ≥ 0.3): 64 B packet, against a median greyscale JPEG crop of 524 B (quality 75) or 451 B (quality 50), and a median original image of 38,323 B.

### How to read these numbers

- The authors' YOLOv4 score is very high, and its class agreement is 100%. Both are consistent with some or all of these images being in its training set. Treat **91.6% as an upper bound**, not as held-out performance.
- **The clean held-out result is our YOLOv8n: 18.8% recall at IoU 0.5.** It is weak: it was trained on only 405 objects from three survey years, and 2018/2021 look different.
- The classical pipeline finds almost nothing at IoU 0.5 and raises 0.775 false alarms per image.
- The oracle row shows the 64-byte packet itself is not the bottleneck: 99.0% of labelled boxes survive encoding at IoU 0.5, with a median centre error of 0.71 px.

## Limitations

- One sonar (Marine Sonic on a Gavia AUV), one dataset, two test survey years.
- Scored with boxes; the dataset has no pixel masks. The outline is not scored separately.
- Pixel → metre conversion uses a nominal 5 cm/pixel; the dataset does not publish resolution.
- CPU times come from one laptop CPU with other software running; they are indicative.
- Not suitable for operational mine countermeasures. This is a compression demo.

## Availability and fallback

The weights are not in git. `authors_yolov4` is read from `data/real/milco/Training/`, and `yolo` from `models/`. When `$MODEL_DIR` is set, both are read from there only. If a model is missing, the evaluation and the website use the next detector (`authors_yolov4` → `yolo` → `classical`) and show a note; tests that need a model are skipped.
