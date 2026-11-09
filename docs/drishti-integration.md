# DRISHTI YOLO11n integration

The frontend and FastAPI routes use a **local trained checkpoint** when it is available. No model binaries or Hugging Face credentials are committed.

## Windows installation

1. From the repository root, double-click `INSTALL_SONAR_AI.bat`.
2. When prompted, authenticate to your own Hugging Face account `ShaarifW`. Credentials stay on your computer.
3. Run `START_HEXADEPTH.bat` (or `python -m uvicorn server:app --reload`).
4. Check [`/api/model/status`](http://localhost:8000/api/model/status): `installed: true`, `runtime_installed: true`. To exercise actual inference, call `/api/analyze`.

## Manual installation (cross-platform)

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-yolo.txt
python -m pip install --upgrade huggingface_hub
python scripts/download_sonar_model.py
python -m uvicorn server:app --reload
```

Default bucket path: `hf://buckets/ShaarifW/hexadepth-checkpoints/hexadepth-v2/run/weights/best.pt`. Saved local path: `models/hexadepth_sonar_best.pt`. Or download manually after `hf auth login`:

```bash
hf buckets cp hf://buckets/ShaarifW/hexadepth-checkpoints/hexadepth-v2/run/weights/best.pt models/hexadepth_sonar_best.pt
```

If a later training run produces better weights, run:

```bash
python scripts/download_sonar_model.py --remote "YOUR/NEW/best.pt" --force
```

The server picks up changed local checkpoint timestamps automatically. You can also set `HEXADEPTH_SONAR_WEIGHTS` to the absolute path of a different locally installed `.pt` checkpoint.

## API

- `GET /api/model/status`: installation and runtime status (does not run inference).
- `POST /api/analyze`: accepts `sample` or image `file`, plus optional `detector_mode=auto|sonar_yolo|classical`. `auto` prefers the installed trained model, then preserves existing fallback. Explicit `sonar_yolo` returns HTTP 503 if the checkpoint/runtime is missing.
- `POST /api/priority`: automatically uses the trained model to pick important regions if installed. `detector`, `detections_total`, and `detected_objects` distinguish model results from the saliency fallback; `regions` still contains up to two prioritized ROIs. Model boxes prioritize the ROI; the transmission remains a software simulation.

Object class IDs in packets are 20 pipeline, 21 shipwreck, 22 ghost net, 23 cylindrical object. These are distinct from legacy MILCO IDs 10 and 11 and classical shape IDs 0–4.

## Model limitations

Checkpoint: YOLO11n, trained 10 epochs at 416 px on DRISHTI (`rehan9599/drishti-sss`). Reported held-out results **within the same DRISHTI dataset**: precision 70.5%, recall 56.4%, mAP@50 60.5%, mAP@50–95 37.7% (450 test images). No independent device/environment validation. The v2 training pipeline excluded `.png` ghost-net examples, so ghost-net recognition is **not validated**. Results are not appropriate for real safety-critical or military operations.

Attribution: DRISHTI dataset is CC BY-SA 4.0 with upstream credits detailed in its dataset card. Checkpoint install reads from a **private** bucket: do not publish the token, checkpoint, or private bucket data.
