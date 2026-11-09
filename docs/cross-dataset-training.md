# Cross-dataset sonar model training and evaluation plan

**Model training is optional for lossless transfer.** Every recognized or unfamiliar sonar image can be delivered exactly with the file-transfer pipeline, provided enough acoustic channel capacity and time. AI is useful for deciding **what to prioritize**, what label to attach and what tiny semantic summary to transmit first. It is **not** a replacement for missing bits if exact image reconstruction is required.

## Define the target domain first

- Avoid mixing fundamentally different sonar modalities as if they were identical color photos. Side-scan sonar (SSS) and forward-looking sonar (FLS) differ in geometry, sampling, noise and available object classes.
- For a Mississippi Sound use case, pick one primary sensor/modality and use cross-location/sensor data for **robustness testing**.
- Common annotation taxonomy: `crab_pot`, `debris`, `pipeline`, `wreck`, `unknown_anomaly` (actual class choices must match ground truth and mission requirements). Do not silently convert unlabeled objects into negative/background examples.

## Candidate public datasets to evaluate for availability, task and license

1. **MILCO**: side-scan imagery for mine-detection research; bundled examples in this project. https://doi.org/10.6084/m9.figshare.24574879
2. **UATD**: forward-looking sonar detection, about 9,200 images; evaluate access, label format and compatibility with your chosen sensor. https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1539371/full
3. **Marine Debris Forward-Looking Sonar Datasets**: tank/turntable/quarry environments; detection, classification and segmentation. https://doi.org/10.5281/zenodo.15101686
4. **SeabedObjects / other SSS datasets**: use the OpenSonarDatasets directory as an index to verify datasets, licensing and modalities: https://github.com/remaro-network/OpenSonarDatasets
5. **China Offshore SSS-AI** (2026): published multi-region side-scan imagery intended to test cross-region generalization; check restrictions and quality before training. https://doi.org/10.5281/zenodo.20048164

## Training + external validation

1. Establish a **baseline** with a compact pretrained detector that runs on your target edge hardware; this repo already has a YOLO dataset preparation script and Colab training notebook. Use actual non-overlapping train/val/test groups, grouped by dive/site/sensor/time to prevent leakage.
2. Normalize sonar dynamic range and coordinates consistently; preserve source resolution and metadata. Adopt sensor-aware speckle/intensity/blur and geometric augmentation that is physically plausible.
3. Train within one modality first; then add harmonized datasets with explicit dataset-aware class masks to avoid conflicting annotations. Include background clutter, hard negatives and true unfamiliar object examples.
4. Reserve **at least one entire dataset or sensor/site** as a never-seen evaluation set. Rotate leave-one-dataset-out tests. Do not choose hyperparameters using the held-out result.
5. Compare per-dataset mAP@50:95, per-class recall and precision, false alarm rate, inference time, confidence calibration (ECE), and *unknown detection/abstention*. A confidence score of 50% is not the same as 50% correctness.
6. Mark uncertain predictions `unknown` and transmit an exact image region or full file instead. Never imply an unfamiliar object was identified with certainty from uncalibrated model output.
7. Separate **recognition** from **reconstruction**. To reconstruct an image exactly, a model-derived approximation must be accompanied by sufficient residual information, or you must use the lossless transfer protocol. A learned generative receiver can produce plausible visual content, but cannot certify the missing physical details as real.

## Appropriate prototype sequence

- Demo A: bit-perfect source-file transport (implemented) on unseen images, measure frame overhead and simulated link performance.
- Demo B: model predicts object bounding boxes/segmentation and sends 64-byte semantic messages (existing detector mode), but adds `unknown` handling and confidence/quality flags.
- Demo C: progressive ROI transmission: metadata immediately, lossless bounding-box crops next, whole original file on demand. Report time to first useful message **and** time to complete file separately.
- Demo D: train on multiple compatible labeled datasets, evaluate held-out datasets, report honest before/after results; target model files and actual external datasets are **not present** in this ZIP and have not been trained here.
