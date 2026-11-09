# Tuning the `real` vision profile (TUNE split only: MILCO 2015, 120 images, 413 objects)

All numbers below were measured on the **TUNE** split (2015 survey). The TEST split (2010, 2017,
2018, 2021) was not looked at during tuning and was evaluated once with the final profile.
Recall ceiling on TUNE with one packet per image: 28.6% (120 images, 413 objects).

Each row adds one change to the row above it unless noted. R = recall, P = precision.

| # | Change | R @ IoU 0.5 | P @ IoU 0.5 | R @ IoU 0.3 | P @ IoU 0.3 | Why |
|---|---|---|---|---|---|---|
| 0 | Synthetic `default` profile | 0.0% | 0.0% | 1.2% | 5.5% | Baseline. Global Otsu splits bright/dark seabed patches into large blobs; the largest wins. |
| 1 | Local-background highlight (31 px) + fixed ratio threshold 1.6 | 0.0% | 0.0% | 0.0% | 0.0% | Targets are small sharp highlights; compare each pixel with its surroundings. Alone it found speckle. |
| 2 | Clean-up kernel 5 → 3 px | 0.0% | 0.0% | 0.0% | 0.0% | A 5 px opening erased small highlights. |
| 3 | Min area 0.08% → 0.01% of image | 0.0% | 0.0% | 0.2% | 0.8% | Targets are ~0.14% of the image; their highlight alone is smaller. |
| 4 | Max area 1% of image | 0.0% | 0.0% | 0.2% | 0.8% | Drop large seabed patches. No effect yet. |
| 5 | Pick highest-contrast blob instead of largest | 0.0% | 0.0% | 0.5% | 1.7% | Largest blob is rarely the target on real data. |
| 6 | Work at full resolution (1024 px) | 0.0% | 0.0% | 0.5% | 1.7% | Avoid shrinking ~30 px targets. Slower. |
| 7 | Bridge drawn track line in nadir mask (6 px) | 0.0% | 0.0% | 0.7% | 2.8% | The dataset draws a yellow line inside the water column, breaking the nadir mask. |
| 8 | Ignore pixels on dark local background (< 0.5 × seabed) | 0.0% | 0.0% | 0.5% | 1.9% | Speckle in the water column had huge contrast against a near-black background. Slightly worse alone. |
| 9 | **Highlight + acoustic shadow pairing** (shadow < 0.5 × background) | 0.7% | 5.2% | 3.6% | 25.9% | Labelled boxes cover highlight **and** shadow; a raised object casts a shadow, speckle does not. |
| 10 | Grid search: background 81 px, shadow < 0.6, pair gap 7, highlight 1.4 | 2.4% | 10.9% | 5.8% | 26.1% | 54-point grid on TUNE. |
| 11 | Bridge up to 20 px in nadir mask | 3.1% | 14.6% | 7.0% | 32.6% | Many false picks were the water-column edge used as a fake "shadow". |
| 12 | Rank by highlight contrast × shadow depth | 4.6% | 21.3% | 10.4% | 48.3% | Strong highlight next to a deep shadow is the classic target signature. |
| 13 | Shadow threshold 0.6 → 0.8 | 7.3% | 31.6% | 12.1% | 52.6% | 72-point grid; captures more of each shadow, so boxes match the labels better. |
| 14 | Pair gap 9 px, dark-background cut 0.3 | **7.7%** | **35.2%** | **12.3%** | **56.0%** | Final `real` profile. |

Tried and dropped (no gain on TUNE): requiring the shadow on the far side of the track from the
highlight; capping elongation at 4 or 6; larger nadir padding (30, 60 px).

## Known gap the TUNE set could not expose

TUNE contains only 2 images without objects (2015 was a dense survey), while TEST has 864 of
1,050 images without objects. The profile always reports its best candidate when one passes the
filters, so on TEST most false alarms come from object-free images (see `real_eval.md`). An
"is there anything here at all" threshold could not be tuned honestly on this TUNE set.
