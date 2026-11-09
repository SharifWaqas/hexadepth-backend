# HexaDepth on real side-scan sonar: d2-test split, `classical` detector, `real` profile

**Detector:** Classical OpenCV pipeline

**Dataset:** MILCO side-scan sonar dataset. Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Side-scan sonar imaging data of underwater vehicles for mine detection. Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132 (data: https://doi.org/10.6084/m9.figshare.24574879)

**Split (Batch D2):** train/val pool = 2010, 2015, 2017 (val = stratified 15% slice, seed 0); test = 2018, 2021. This run: **d2-test** (2018, 2021).

**Images:** 612 (139 with objects, 473 without) · sizes {'1024x1024': 255, '416x416': 357} · ground-truth objects 191 {'MILCO': 145, 'NOMBO': 46} · degenerate boxes 2

**Date:** 2026-11-07T20:44:01-06:00 · **CPU:** Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (8 threads) · Python 3.14.8 · OpenCV 4.14.0

## Detection

| IoU ≥ | Recall | Precision | TP / GT | Predictions | False alarms / image | MILCO recall | NOMBO recall | Median centre error |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5% | 0.2% | 1 / 191 | 475 | 0.775 | 0.0% (0/145) | 2.2% (1/46) | 4.47 px |
| 0.3 | 5.2% | 2.1% | 10 / 191 | 475 | 0.76 | 4.1% (6/145) | 8.7% (4/46) | 5.98 px |

False alarms at IoU ≥ 0.5: 364 on the 473 images with no object, 110 on the 139 images with objects.
False alarms at IoU ≥ 0.3: 364 on the 473 images with no object, 101 on the 139 images with objects.

Recall ceiling for this detector: 72.8%. Images with a detection: 475 of 612.

## Speed

Per image (detector + pack + unpack): median 54.9 ms, 95th percentile 438.33 ms.

## Size: what would be sent for each matched object

Matched objects (IoU ≥ 0.3): 10

| Payload | Median bytes | HexaDepth reduction | Send time at 100 bps |
|---|---|---|---|
| HexaDepth packet | 64 | n/a | 5.12 s |
| JPEG crop, quality 75 | 642.0 | 90.03% | 51.36 s |
| JPEG crop, quality 50 | 535.5 | 88.05% | 42.84 s |
| Full original .jpg | 185,959.5 | 99.97% | 14876.76 s |

## Packets

Produced 475 · exactly 64 bytes 100.0% · valid CRC on decode 100.0% · pack errors 0

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

- One packet per image: the classical detector reports at most one object per image, so recall cannot exceed recall_upper_bound_pct.
- Box IoU: bounding box of the polygon rebuilt from the decoded 64 bytes vs the ground-truth YOLO box. The dataset has no pixel masks.
- Ground-truth boxes with zero width or height are kept and count as misses.
- Pixel positions are decoded with the same nominal metres-per-pixel used to encode; pixel metrics do not depend on its true value.
- Size comparison uses objects matched at IoU >= 0.3; crops are the ground-truth box + 10% padding, encoded as greyscale JPEG (smaller than colour, so the comparison is the least favourable to HexaDepth).
- Speed: detector + pack + unpack per image, single process, excludes file reading.
- Class agreement: share of matched objects whose decoded class (MILCO/NOMBO) equals the label; only for detectors that predict the dataset classes.
