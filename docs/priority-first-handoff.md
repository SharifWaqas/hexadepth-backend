# HexaDepth: priority-first, progressive underwater information delivery

## The actual user problem

A submerged vehicle cannot quickly send a large sonar file through a narrowband underwater acoustic modem. The operator wants an early indication of possible anomalies, followed by useful scene context and enhanced image regions, without waiting for the exact file. False AI labels are undesirable, especially on unseen sensors or datasets.

## Implemented in this version

1. **ROI metadata first.** A compact binary message carries width, height, up to two regions, a heuristic score, optional class ID, and a trained-detector flag. For two regions it is **30 bytes**, fitting into one **64-byte frame** (46-byte payload capacity). The display says *unverified region* unless a separately installed trained model supplies predictions; even then predictions are not ground truth.
2. **Whole-scene overview second.** The vehicle downscales the grayscale image to 112 pixels across its longer dimension and JPEG-encodes it. Encoded payload bytes are broken into actual fixed 64-byte frames, passed through CRC-16 validation and reassembled. Receiver can render an approximation even if no object was recognized.
3. **Prioritized detail third.** Up to two salient/AI-detected regions are JPEG encoded separately (longest crop dimension 176 pixels, JPEG quality 42). Each crop includes its ROI index and coordinates in the actual transmitted payload. Receiver decodes the validated bytes and overlays the patches onto the overview. This is **lossy**; details outside regions remain coarse.
4. **Original image on request.** Press **Request exact image** to invoke the existing, independent `/api/lossless` pipeline. All source file bytes are sent through CRC-framed 64-byte packets and verified by SHA-256 at the receiving end. No data is omitted when exact fidelity is required.
5. **Failure handling.** Corrupted frames and invalid metadata are rejected; no false intact image is displayed on failed integrity checks. The backend exposes `corrupt_stage=metadata|overview|detail` for error tests.

### Model options

- The supplied source **does not contain trained YOLO weights**. In the default demo, OpenCV gradient and local-contrast saliency select visually interesting rectangles; this is **not AI semantic classification**. On uniform imagery, the pipeline abstains from reporting any regions rather than inventing objects.
- If `models/yolo_milco_v8n.pt` is installed together with the optional Ultralytics dependencies, `/api/priority` first attempts that trained detector for region selection. The authors' YOLOv4 model files are another option. Trained boxes are combined with saliency for any remaining regions.
- Neither included nor optional weights prove performance on a new sensor or unseen dataset. Labels must be presented as predictions with an *unknown* option, and independently evaluated across datasets/sites/sensors before deployment.

## What the benchmark actually demonstrates

Run `python scripts/benchmark_priority.py`. It measures on four included real MILCO sonar JPEG examples: encoded pixel data and metadata bytes, actual 64-byte frame counts, CRC-validated reassembly, and idealized airtime at 2,400 bps. It estimates how long before a **coarse** scene is visible vs how long it would take to transfer the full original exactly.

This is not a proof of faster transfer **at the same quality**: the early representation discards pixel details. The reported speedup is **time-to-first-rough-view** only. The modem airtime excludes preambles, FEC, contention, dropouts/retries and hardware and encoding latency. The 0.48-second fixed setup/propagation delay is a demonstration constant, not a real physical measurement.

## Why we didn't claim to train a generalizable AI

There are no bundled multiple full labeled datasets or model checkpoints. Four representative public sonar images are not enough to train or validate a broadly generalizable detector. The Colab notebook in `notebooks/` is an earlier single-dataset training recipe, not a new cross-domain model.

A rigorous next phase is:

1. Select one primary sonar modality and target sensor; do not silently mix incompatible FLS/SSS categories.
2. Collect licensed, labeled data from multiple **different sites and sensors**. Harmonize labels and capture unknown/negative examples.
3. Train a detector (e.g. YOLO) on site-grouped splits, and calibrate its confidence with real validation data.
4. Test on **entire held-out datasets** and unseen sensors. Report per-class precision/recall, mAP, false-positive rate, latency, and unknown-object/abstention behavior. Never test on training or augmented copies.
5. Measure whether ROI selection improves **task utility**: did the operator receive the actual needed object soon enough and with enough detail? Benchmark generic saliency and random ROI selection as baselines.
6. Integrate a supported acoustic modem/bridge and measure achieved bit rate, retransmits, packet loss and end-to-end latency; use the modem's actual frame MTU and MAC/FEC rather than assuming 64 bytes are physically optimal.

## Running the code

Extract the ZIP, open its containing folder (which should have `server.py`, `requirements.txt`, and `START_HEXADEPTH.bat`). On Windows, double-click the `.bat` file, or in Command Prompt:

```bat
cd /d "%USERPROFILE%\Downloads\HexaDepth_Priority_First"
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
```

Open http://localhost:8000. The new priority-first dashboard is the default page. The earlier interface is retained at http://localhost:8000/transport.html.

## Interface and API

- `POST /api/priority`, `multipart/form-data`: `sample` or `file`, `rate_bps` (default 2400), optional `corrupt_stage`. Returns measured wire bytes, framed packet counts, reconstructed images as PNG data URIs, confidence provenance, and simulation times. **HTTP itself is not the underwater transport:** the server runs the packetizer/receiver in-process to verify the bytes.
- `POST /api/lossless`: exact original bytes on request, CRC-checked 64-byte frames and SHA-256.
- `POST /api/analyze`: legacy detector mode.
- `POST /api/preview`: legacy 8-byte binary sketch.

## Technical limits / research honesty

This demo is **not** a trained image-to-image neural decoder. The reconstructed preview uses JPEG and ROI patch overlays, not AI-generated missing details. We deliberately do not hallucinate missing sonar evidence. It is also **not** field-tested on a military or commercial underwater modem and does not connect to any military system.

Attribution: bundled real sonar examples from MILCO, Santos et al. (2024), CC BY 4.0; see `web/samples/real/README.md`.
