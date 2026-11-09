# HexaDepth on real side-scan sonar: detector comparison

> **Held-out set:** No held-out set could be built for the authors' YOLOv4: the dataset does not include the authors' training/validation image lists. All detectors are evaluated on the Batch D2 TEST years (2018 + 2021), which our YOLOv8n and the classical pipeline never saw. Authors' YOLOv4 numbers may include images the model was trained on: the authors did not publish which images they trained and validated on (their notebook builds train.txt / test.txt on Google Drive; the lists are not in the dataset).

**Dataset:** MILCO side-scan sonar dataset. Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Side-scan sonar imaging data of underwater vehicles for mine detection. Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132 (data: https://doi.org/10.6084/m9.figshare.24574879, CC BY 4.0)

**Images:** 612 from 2018, 2021 (139 with objects, 473 without) · ground-truth objects 191 {'MILCO': 145, 'NOMBO': 46} · degenerate boxes 2

**Rules:** Same images, same code path for every detector: detector -> 64-byte packet -> decode from the bytes only -> bounding box of the decoded outline vs the labelled box; IoU 0.5 and 0.3; greedy one-to-one matching; every image counted, including images with no objects.

**Date:** 2026-11-07T20:44:05-06:00 · **CPU:** Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (8 threads) · Python 3.14.8 · OpenCV 4.14.0

## Detection at IoU ≥ 0.5

| Detector | Trained on these images? | Recall | Precision | False alarms / image | MILCO recall | NOMBO recall | Class agreement |
|---|---|---|---|---|---|---|---|
| YOLOv4 trained by Santos et al. (2024) | Unknown: may include training images | 91.6% (175/191) | 91.6% | 0.026 | 91.0% (132/145) | 93.5% (43/46) | 100.0% |
| YOLOv8n fine-tuned by HexaDepth (D2 split) | No | 18.8% (36/191) | 29.3% | 0.142 | 18.6% (27/145) | 19.6% (9/46) | 83.3% |
| Classical OpenCV pipeline | No | 0.5% (1/191) | 0.2% | 0.775 | 0.0% (0/145) | 2.2% (1/46) | n/a |
| Oracle (ground-truth boxes) | n/a | 99.0% (189/191) | 99.0% | 0.003 | 98.6% (143/145) | 100.0% (46/46) | n/a |

## Detection at IoU ≥ 0.3

| Detector | Trained on these images? | Recall | Precision | False alarms / image | MILCO recall | NOMBO recall | Class agreement |
|---|---|---|---|---|---|---|---|
| YOLOv4 trained by Santos et al. (2024) | Unknown: may include training images | 94.8% (181/191) | 94.8% | 0.016 | 94.5% (137/145) | 95.7% (44/46) | 100.0% |
| YOLOv8n fine-tuned by HexaDepth (D2 split) | No | 22.0% (42/191) | 34.1% | 0.132 | 22.8% (33/145) | 19.6% (9/46) | 83.3% |
| Classical OpenCV pipeline | No | 5.2% (10/191) | 2.1% | 0.76 | 4.1% (6/145) | 8.7% (4/46) | n/a |
| Oracle (ground-truth boxes) | n/a | 99.0% (189/191) | 99.0% | 0.003 | 98.6% (143/145) | 100.0% (46/46) | n/a |

## CPU time per image (detector + pack + unpack)

| Detector | Median | 95th percentile |
|---|---|---|
| YOLOv4 trained by Santos et al. (2024) | 798.68 ms | 1059.88 ms |
| YOLOv8n fine-tuned by HexaDepth (D2 split) | 237.58 ms | 348.39 ms |
| Classical OpenCV pipeline | 54.9 ms | 438.33 ms |
| Oracle (ground-truth boxes) | 0.0 ms | 0.09 ms |

## What gets sent per matched object (IoU ≥ 0.3)

| Detector | Matched objects | HexaDepth packet | JPEG crop q75 (median) | JPEG crop q50 (median) | Full image (median) |
|---|---|---|---|---|---|
| YOLOv4 trained by Santos et al. (2024) | 181 | 64 B | 524.0 B (87.79% smaller as 64 B) | 451.0 B (85.81% smaller as 64 B) | 38,323.0 B (99.83% smaller as 64 B) |
| YOLOv8n fine-tuned by HexaDepth (D2 split) | 42 | 64 B | 482.5 B (86.74% smaller as 64 B) | 430.5 B (85.13% smaller as 64 B) | 37,138.5 B (99.83% smaller as 64 B) |
| Classical OpenCV pipeline | 10 | 64 B | 642.0 B (90.03% smaller as 64 B) | 535.5 B (88.05% smaller as 64 B) | 185,959.5 B (99.97% smaller as 64 B) |
| Oracle (ground-truth boxes) | 189 | 64 B | 524.0 B (87.79% smaller as 64 B) | 451.0 B (85.81% smaller as 64 B) | 38,444.0 B (99.83% smaller as 64 B) |

## Packets

- YOLOv4 trained by Santos et al. (2024): 191 packets, exactly 64 bytes 100.0%, valid CRC 100.0%, pack errors 0
- YOLOv8n fine-tuned by HexaDepth (D2 split): 123 packets, exactly 64 bytes 100.0%, valid CRC 100.0%, pack errors 0
- Classical OpenCV pipeline: 475 packets, exactly 64 bytes 100.0%, valid CRC 100.0%, pack errors 0
- Oracle (ground-truth boxes): 191 packets, exactly 64 bytes 100.0%, valid CRC 100.0%, pack errors 0

## Credits

- Detector: YOLOv4 trained by Santos et al. (2024), CC BY 4.0 (https://doi.org/10.6084/m9.figshare.24574879).
- Dataset: Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Side-scan sonar imaging data of underwater vehicles for mine detection. Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132; data https://doi.org/10.6084/m9.figshare.24574879 (CC BY 4.0).

Per-detector details: `results/real_eval_d2-test_<detector>.md`.
