# HexaDepth on real side-scan sonar: test split, `classical` detector, `default` profile

**Dataset:** MILCO side-scan sonar dataset. Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). Side-scan sonar imaging data of underwater vehicles for mine detection. Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132 (data: https://doi.org/10.6084/m9.figshare.24574879)

**Split:** tune = 2015; test = 2010, 2017, 2018, 2021. This run: **test** (2010, 2017, 2018, 2021).

**Images:** 1050 (186 with objects, 864 without) · sizes {'416x416': 535, '1024x1024': 515} · ground-truth objects 255 {'NOMBO': 60, 'MILCO': 195} · degenerate boxes 2

**Date:** 2026-11-07T18:47:39-06:00 · **CPU:** Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (8 threads) · Python 3.14.8 · OpenCV 5.0.0

## Detection

| IoU ≥ | Recall | Precision | TP / GT | Predictions | False alarms / image | MILCO recall | NOMBO recall | Median centre error |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0% | 0.0% | 0 / 255 | 507 | 0.483 | 0.0% (0/195) | 0.0% (0/60) | n/a |
| 0.3 | 0.4% | 0.2% | 1 / 255 | 507 | 0.482 | 0.0% (0/195) | 1.7% (1/60) | 57.29 px |

False alarms at IoU ≥ 0.5: 402 on the 864 images with no object, 105 on the 186 images with objects.
False alarms at IoU ≥ 0.3: 402 on the 864 images with no object, 104 on the 186 images with objects.

Recall ceiling for this detector: 72.9%. Images with a detection: 507 of 1050.

## Speed

Per image (detector + pack + unpack): median 43.24 ms, 95th percentile 91.24 ms.

## Size: what would be sent for each matched object

Matched objects (IoU ≥ 0.3): 1

| Payload | Median bytes | HexaDepth reduction | Send time at 100 bps |
|---|---|---|---|
| HexaDepth packet | 64 | n/a | 5.12 s |
| JPEG crop, quality 75 | 7,708.0 | 99.17% | 616.64 s |
| JPEG crop, quality 50 | 5,331.0 | 98.8% | 426.48 s |
| Full original .jpg | 391,384.0 | 99.98% | 31310.72 s |

## Packets

Produced 507 · exactly 64 bytes 100.0% · valid CRC on decode 100.0% · pack errors 0

## Vision parameters

```json
{
  "meters_per_pixel": 0.05,
  "work_width": 800,
  "median_ksize": 5,
  "bilateral_d": 9,
  "bilateral_sigma": 60.0,
  "min_contrast": 1.5,
  "mask_nadir": true,
  "min_area_frac": 0.0008,
  "epsilon_frac": 0.02,
  "max_vertices": 20,
  "local_bg_ksize": 0,
  "threshold_ratio": 0.0,
  "morph_ksize": 5,
  "max_area_frac": 1.0,
  "select": "area",
  "nadir_gap": 0,
  "nadir_pad": 12,
  "min_bg_frac": 0.0,
  "shadow_ratio": 0.0,
  "pair_gap": 5,
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
