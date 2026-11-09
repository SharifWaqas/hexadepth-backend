# HexaDepth

**Model-independent visual previews and detector-assisted object packets for constrained underwater acoustic communications.**

HexaDepth now has **two distinct encodings** on its simplified web dashboard:

| Mode | Payload | What the receiver gets | Needs trained AI? |
|---|---:|---|---|
| **Visual preview (default)** | **64 bits = 8 bytes** | An 8×8 binary light/dark grid from *any* grayscale sonar frame | **No** |
| Object packet (legacy) | **64 bytes = 512 bits** | A detected outline, approximate size and position, and class estimate | Detection-dependent |

**Important:** The 8-byte mode is an intentionally crude semantic/visual preview, **not** a recognizable photograph or robust object detector. At this size, the original image cannot be recovered. Its eight payload bytes contain no coordinate metadata or error checksum. The link simulator adds fragmentation and CRC overhead *on the wire*; do not confuse payload size with actual transmitted bytes.

**Transport:** The browser demo simulates a 1–4 lane acoustic link at configurable bit rates. Lanes represent independent simultaneous acoustic channels/modems; multithreading one acoustic modem is **not** extra physical bandwidth. The model includes 5 bytes of framing per fragment, illustrative 0.15 s setup and 0.33 s propagation, but **no** modem synchronization, acoustic interference, throughput adaptation, FEC, field measurements, or physical device. **Sonar acquires the image; a compatible acoustic data modem transmits the bytes.** This is a research demonstration, not a tested underwater communications link.

## Quick start on Windows

After extracting **HexaDepth_Updated_Prototype.zip** using Windows **Extract All**, open Command Prompt in the extracted directory. This updated ZIP places `server.py` and `requirements.txt` at the top level (no confusing extra nested project directory):

```bat
cd "C:\Users\sarim\Downloads\HexaDepth_Updated_Prototype"
dir
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
```

Open **http://localhost:8000**. Choose a sonar sample (including bundled real data), and press **Create preview**. Switch to *64-byte object packet* to demonstrate the legacy detector. Choose 1, 2, or 4 independent simulated acoustic channels and optionally inject a bit error to verify CRC rejection.

If Windows extracted into a different folder, change the `cd` path accordingly. In all cases, `dir` should list `server.py`, `requirements.txt`, and `web` directly.

### API endpoints

- `POST /api/preview` multipart `sample=<name>` or `file=<image>` — 8-byte model-free image preview and reconstructed 8×8 binary grid.
- `POST /api/link/simulate` JSON `{"payload_hex":"...","channels":1,"rate_bps":100,"corrupt":false,"raw_bytes":480000}` — framed fragment transfer, reassembly and CRC checks with independent-link airtime estimates.
- `POST /api/analyze`, `/api/transmit`, `/api/decode` — original 64-byte detector workflow remains available and functional.

Test with `python -m pytest -q`.

### Legacy Streamlit view

The earlier Streamlit demo remains available with `python -m streamlit run app.py`. It still focuses on the 64-*byte* detector packet, not the new 64-*bit* model-independent visual preview.

## Legacy Streamlit setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Five synthetic sonar frames are generated into `samples/` automatically on first run.

## Web server setup (macOS / Linux)

The web dashboard (FastAPI + plain HTML/CSS/JS, no build step) uses the same Python modules:

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

Then open http://localhost:8000. Press **Create preview** to run model-free encode → simulate link → reconstruct in one click. Switch modes to try the original detection-based object packet.
The Streamlit dashboard above (`streamlit run app.py`) still works as a backup.

Run the tests:

```bash
python -m pytest -q
```

Run the vision pipeline on a folder of images (Developer 1's deliverable):

```bash
python -m hexadepth.vision samples/*.png
```

## Real data evaluation (MILCO side-scan dataset)

HexaDepth is measured on real side-scan sonar from the MILCO dataset: Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). *Side-scan sonar imaging data of underwater vehicles for mine detection.* Data in Brief, 53, 110132. https://doi.org/10.1016/j.dib.2024.110132. Data: https://doi.org/10.6084/m9.figshare.24574879

The images are **not** in git. Download them and unpack so the layout is `data/real/milco/<year>/<year>/*.jpg` (+ matching `.txt` YOLO labels) and `data/real/milco/Training/obj.names.txt`.

```bash
python scripts/evaluate_real.py --compare                                        # headline -> results/real_eval.{json,md}
python scripts/evaluate_real.py --split d2-test --detector authors_yolov4         # one detector
python scripts/evaluate_real.py --split test --detector classical --profile real  # Batch D run (tune 2015 / test 4 years)
python scripts/evaluate_real.py --split tune --profile real --limit 20            # quick run
```

- **Headline (`--compare`):** every detector on the same 612 images from 2018 + 2021 with identical rules (detector → 64-byte packet → decode from the bytes only → IoU 0.5 and 0.3, one-to-one matching, all images counted). Results: `results/real_eval.md`; details per detector: `results/real_eval_d2-test_<detector>.md`.
- **Detectors** (`hexadepth/detectors.py`, shared with the web server):
  - `authors_yolov4`: the dataset authors' trained YOLOv4 (Santos et al. 2024, CC BY 4.0), run unchanged on CPU with OpenCV DNN. **The authors did not publish their training/validation image lists, so its numbers may include images the model was trained on.** Confidence threshold 0.25 (Darknet default), NMS 0.45.
  - `yolo`: our YOLOv8n fine-tuned on the D2 train split (Colab notebook below). Needs `requirements-yolo.txt`.
  - `classical`: the OpenCV pipeline with a named vision profile (`real` tuned on 2015 only).
  - `oracle`: the labelled boxes as detections; shows what the 64-byte packet preserves.
  - Box detectors send one packet per detected object: the outline is extracted inside each box (`vision.outline_in_box`) and fitted to it.
- **Model files are not in git** (see `models/README.md`): the YOLOv4 weights are read from `data/real/milco/Training/`, the YOLOv8n from `models/yolo_milco_v8n.pt`, or both from `$MODEL_DIR`. If one is missing, the evaluation and the website fall back to the next detector (`authors_yolov4` → `yolo` → `classical`) with a visible note; tests that need a model skip.
- **OpenCV 4.x is required** (`<5`): OpenCV 5.0 removed the Darknet importer.
- `results/real_tuning.md` logs every tuning step of the classical `real` profile; `results/real_eval_test_*.md` are the Batch D results (classical tuned on 2015, tested on four years).
- The web dashboard's **Results** section reads `results/real_eval.json` via `GET /api/results`. Real samples are analysed with the best available trained detector; synthetic samples and uploads use the classical pipeline. Model card: `docs/model_card.md`.

## Trained detector (Batch D2)

Split for the trained detector: **train/val pool = 2010 + 2015 + 2017** (val = stratified ~15% random slice by survey year × image type, seed 0), **test = 2018 + 2021** (`--split d2-train|d2-val|d2-test` in `scripts/evaluate_real.py`; the Batch D `tune`/`test` split still works).

```bash
python scripts/make_yolo_dataset.py        # -> build/yolo_milco/ and build/yolo_milco.zip (TRAIN + VAL only)
```

Then train on Google Colab with `notebooks/train_yolo_colab.ipynb` (GPU runtime), download `best.pt` and save it as `models/yolo_milco_v8n.pt`.

The four real sample images in `web/samples/real/` are from the MILCO dataset (CC BY 4.0); see `web/samples/real/README.md` for attribution. Set `HEXADEPTH_REAL_DATA` to point the server and tests at a different dataset location.

## Project layout

| File | Module | What it does |
|---|---|---|
| `hexadepth/vision.py` | 1 · Edge vision | Denoise → range-gain normalise → Otsu → contour → RDP polygon (≤ 20 vertices) → 0–255 grid |
| `hexadepth/packet.py` | 2 · Bit-packer | Exactly 64-byte binary packet with CRC-16, hex encode/decode |
| `hexadepth/topside.py` | 3 · Reconstruction | Draws the decoded target on a sonar-style screen |
| `app.py` | 3 · Dashboard | Streamlit UI: drone → acoustic link → ship (backup) |
| `server.py` + `web/` | 3 · Web dashboard | FastAPI API over the modules + HTML/CSS/JS frontend |
| `hexadepth/synth.py` | Test data | Synthetic side-scan frames (speckle, range falloff, nadir, shadows) |
| `hexadepth/detectors.py` | Detectors | classical, oracle, authors' YOLOv4 (OpenCV DNN), fine-tuned YOLOv8n; fallbacks |
| `hexadepth/realdata.py` + `scripts/evaluate_real.py` | Evaluation | Real-data benchmark: YOLO labels, box IoU, matching, size comparison |
| `tests/` | QA | Lossless round-trip, CRC catches every single-bit error, ≤ 20 vertices, size/position/class accuracy |

## The 64-byte packet

| Offset | Type | Field | Units / range |
|---|---|---|---|
| 0 | `B` | Protocol version | 1 |
| 1 | `H` | UUV ID | 0–65535 |
| 3 | `I` | Timestamp | Unix seconds |
| 7 | `H` | Vehicle easting | metres from local datum, 0–65 535 |
| 9 | `H` | Vehicle northing | metres from local datum, 0–65 535 |
| 11 | `H` | Vehicle depth | decimetres, 0–6 553.5 m |
| 13 | `h` | Target offset east of vehicle | decimetres, ±3 276.7 m |
| 15 | `h` | Target offset north of vehicle | decimetres, ±3 276.7 m |
| 17 | `H` | Target extent (longest side) | centimetres, 0–655.35 m |
| 19 | `B` | Class ID | see `CLASS_NAMES` |
| 20 | `B` | Confidence | 0–100 |
| 21 | `B` | Vertex count | 3–20 |
| 22 | `40B` | 20 × (x, y) vertices | 0–255 grid, unused pairs = 0 |
| 62 | `H` | CRC-16/CCITT-FALSE | over bytes 0–61 |
| | | **Total** | **64 bytes** (`struct` format `>BHIHHHhhHBBB40BH`) |

Both axes use one scale factor (longest side → 255), so shapes keep their true proportions. `extent` converts grid units back to metres.

## Historical 64-byte pitch numbers (100 bps, payload airtime only)

| | Size | Bits | Time at 100 bps |
|---|---|---|---|
| Raw image | 2,000,000 bytes | 16,000,000 | 160,000 s ≈ **44.4 hours** |
| HexaDepth | 64 bytes | 512 | **5.12 seconds** |

Reduction 99.9968% · 31,250× less data.

## Legacy detector demo script

1. Pick a sample in column 1 and show the raw frame → threshold → green polygon, then the hex packet.
2. In column 2, move the baud slider and point out that the raw image still takes hours at 2,400 bps.
3. Press **Transmit**. Column 3 rebuilds the target from the hex string alone.
4. Set **bit errors** to 1 and transmit again. The CRC rejects the packet, so topside never acts on corrupted data.
5. Edit one character of the received hex by hand to show the same thing.

## Known simplifications (be upfront if judges ask)

- **Classifier:** on synthetic data the classical pipeline's classes are rule-based (elongation, circularity, fill). On real samples the trained detector reports the dataset's MILCO / NOMBO classes (packet class ids 10 / 11; `packet.py` is unchanged).
- **Authors' YOLOv4:** its training images are unknown, so its scores on the MILCO images may be optimistic. Our YOLOv8n is the clean held-out comparison.
- **Geometry:** the vehicle is assumed at the image centre heading north. A real system uses navigation data (heading, altitude, slant-range correction).
- **Input:** real side-scan arrives ping by ping. A real system buffers pings into tiles before running the pipeline.
- **Shadows** help find objects in the classical `real` profile but are not turned into height yet. Shadow length gives object height, a strong next feature.
- **Modem command:** not implemented. Check your modem's manual for the correct data-packet sentence; don't put an unverified command on a slide.
- **Synthetic data** is cleaner than real sonar. Test with real side-scan samples and retune the sidebar "Vision tuning" sliders.
