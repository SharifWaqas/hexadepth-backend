# HexaDepth — Priority-first sonar transmission backend

A hackathon research prototype that sends useful sonar information first: object-region coordinates, a low-resolution whole-scene overview, then prioritized detail crops. It also supports optional **byte-exact full-image recovery**.

This repository includes a **FastAPI backend**, Python packet/transport modules, browser demo, benchmarks, and automated tests.

> **Important:** Underwater acoustic transfer times are **software simulations** based on selected bit rates and public modem specifications. No real acoustic modem or underwater hardware has been tested. Current default visual-region selection is not a validated object-recognition model; the optional trained DRISHTI YOLO11n checkpoint is **not included** and must be installed locally.

## Quick start (Windows / macOS / Linux)

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
```

- Demo: <http://localhost:8000/priority.html>
- Interactive API specification: <http://localhost:8000/docs>
- Legacy transport demo: <http://localhost:8000/transport.html>
- API handoff: [`FRONTEND_HANDOFF.md`](FRONTEND_HANDOFF.md)

Windows users can also double-click `START_HEXADEPTH.bat`.

## Core features

- **Priority-first delivery:** metadata/coordinates → lossy whole-scene overview → lossy object-region detail → optional exact file.
- **64-byte software packet frames** with CRC-16 validation and receiver-side reassembly.
- **Lossless recovery** mode with SHA-256 integrity verification.
- **Simulated transfer-time comparison** against full-file delivery at the same selected link speed.
- **Multiple detection modes**, with fallback when model weights are absent.
- **Benchmark scripts** and tested API endpoints.

## Trained sonar AI (optional)

Run `INSTALL_SONAR_AI.bat` once on Windows (or follow [DRISHTI setup](docs/drishti-integration.md)) to install the private `best.pt` weights locally; then start/restart FastAPI. `GET /api/model/status` reports installation. Both `/api/analyze` and `/api/priority` prefer the trained YOLO11n detector when installed, and expose class names, boxes, and confidences. Default fallback remains available without model weights. v2 achieved 70.5% precision/56.4% recall on a same-dataset DRISHTI test set; performance on unseen sonar sources is unknown.

## For frontend developers

The primary endpoint is `POST /api/priority` using `multipart/form-data`, with a `sample` name or uploaded `file`, optionally `rate_bps`. `GET /api/samples` lists available demo images. `POST /api/lossless` exercises exact-image recovery. See [`FRONTEND_HANDOFF.md`](FRONTEND_HANDOFF.md) and `/docs` for current parameters and response schemas.

## Tests

```bash
python -m pytest -q
```

Verified on the prepared source: **324 passed, 4 skipped**. Skips cover optional models/data not distributed with this repository.

## Repository structure

| Path | Purpose |
| --- | --- |
| `server.py` | FastAPI endpoints |
| `hexadepth/priority.py` | Progressive payload generation |
| `hexadepth/packet.py` | Fixed-size detection packet encoding |
| `hexadepth/file_transport.py` | Full-file packetization and recovery |
| `hexadepth/link.py` | Modeled link transport |
| `hexadepth/comparison.py` | Transfer-time comparison |
| `web/` | Browser demo and sample assets |
| `tests/` | Automated tests |
| `scripts/` | Reproducible benchmarks/evaluation |
| `docs/` | Design and methodology |
| `results/` | Existing benchmark output |

## Scope and limitations

- No real acoustic communications hardware is controlled by this code.
- No military equipment or operational mine-detection system was tested.
- The lossless option preserves the original file, but is **not** faster merely because it uses smaller packet frames.
- The trained YOLO11n DRISHTI model is supported via local checkpoint installation; its weights are not distributed in this repository.
- Multi-dataset generalization has **not** been demonstrated.

## Sample data attribution

Bundled side-scan sonar examples are credited to Santos et al. (2024), *Side-scan sonar imaging data of underwater vehicles for mine detection*, [DOI: 10.6084/m9.figshare.24574879](https://doi.org/10.6084/m9.figshare.24574879), licensed CC BY 4.0. The package also generates synthetic demonstration scenes.

See [`docs/model_card.md`](docs/model_card.md) and [`docs/priority-first-handoff.md`](docs/priority-first-handoff.md) for caveats and technical details.
