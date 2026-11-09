# HexaDepth on real side-scan sonar: d2-test split, `oracle` detector

> **Oracle mode:** ground-truth boxes are used as detections. These numbers show what the 64-byte packet preserves, not how well objects are found.

**Detector:** Oracle (ground-truth boxes)

**Dataset:** MILCO side-scan sonar dataset. Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Side-scan sonar imaging data of underwater vehicles for mine detection. Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132 (data: https://doi.org/10.6084/m9.figshare.24574879)

**Split (Batch D2):** train/val pool = 2010, 2015, 2017 (val = stratified 15% slice, seed 0); test = 2018, 2021. This run: **d2-test** (2018, 2021).

**Images:** 612 (139 with objects, 473 without) · sizes {'1024x1024': 255, '416x416': 357} · ground-truth objects 191 {'MILCO': 145, 'NOMBO': 46} · degenerate boxes 2

**Date:** 2026-11-07T20:44:05-06:00 · **CPU:** Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (8 threads) · Python 3.14.8 · OpenCV 4.14.0

## Detection

| IoU ≥ | Recall | Precision | TP / GT | Predictions | False alarms / image | MILCO recall | NOMBO recall | Median centre error |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 99.0% | 99.0% | 189 / 191 | 191 | 0.003 | 98.6% (143/145) | 100.0% (46/46) | 0.71 px |
| 0.3 | 99.0% | 99.0% | 189 / 191 | 191 | 0.003 | 98.6% (143/145) | 100.0% (46/46) | 0.71 px |

False alarms at IoU ≥ 0.5: 0 on the 473 images with no object, 2 on the 139 images with objects.
False alarms at IoU ≥ 0.3: 0 on the 473 images with no object, 2 on the 139 images with objects.

Recall ceiling for this detector: 100.0%. Images with a detection: 139 of 612.

## Speed

Per image (detector + pack + unpack): median 0.0 ms, 95th percentile 0.09 ms.

## Size: what would be sent for each matched object

Matched objects (IoU ≥ 0.3): 189

| Payload | Median bytes | HexaDepth reduction | Send time at 100 bps |
|---|---|---|---|
| HexaDepth packet | 64 | n/a | 5.12 s |
| JPEG crop, quality 75 | 524.0 | 87.79% | 41.92 s |
| JPEG crop, quality 50 | 451.0 | 85.81% | 36.08 s |
| Full original .jpg | 38,444.0 | 99.83% | 3075.52 s |

## Packets

Produced 191 · exactly 64 bytes 100.0% · valid CRC on decode 100.0% · pack errors 0

## Vision parameters

```json
{
  "meters_per_pixel": 0.05,
  "work_width": 1024,
  "median_ksize": 5,
  "bilateral_d": 9,
  "bilateral_sigma": 60.0,
  "min_contrast": 1.5,
  "mask_nadir": true,
  "min_area_frac": 0.0001,
  "epsilon_frac": 0.02,
  "max_vertices": 20,
  "local_bg_ksize": 81,
  "threshold_ratio": 1.4,
  "morph_ksize": 3,
  "max_area_frac": 0.01,
  "select": "pair",
  "nadir_gap": 20,
  "nadir_pad": 12,
  "min_bg_frac": 0.3,
  "shadow_ratio": 0.8,
  "pair_gap": 9,
  "min_shadow_frac": 0.15
}
```

## Notes

- ORACLE MODE: ground-truth boxes are used as detections. These numbers measure only what the 64-byte packet preserves (centre, size, outline); they are NOT detection results.
- Box IoU: bounding box of the polygon rebuilt from the decoded 64 bytes vs the ground-truth YOLO box. The dataset has no pixel masks.
- Ground-truth boxes with zero width or height are kept and count as misses.
- Pixel positions are decoded with the same nominal metres-per-pixel used to encode; pixel metrics do not depend on its true value.
- Size comparison uses objects matched at IoU >= 0.3; crops are the ground-truth box + 10% padding, encoded as greyscale JPEG (smaller than colour, so the comparison is the least favourable to HexaDepth).
- Speed: detector + pack + unpack per image, single process, excludes file reading.
- Class agreement: share of matched objects whose decoded class (MILCO/NOMBO) equals the label; only for detectors that predict the dataset classes.
