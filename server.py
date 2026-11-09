"""
HexaDepth web backend (FastAPI).

Thin HTTP layer over the existing modules: hexadepth.vision (detect), hexadepth.packet
(64-byte encode / decode / CRC) and hexadepth.synth (sample data). No algorithm lives here.

Run:  uvicorn server:app --reload     then open http://localhost:8000
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexadepth import detectors as dets
from hexadepth import packet as pk
from hexadepth import preview as pv
from hexadepth import link as ln
from hexadepth import file_transport as ft
from hexadepth import priority as pr
from hexadepth import synth
from hexadepth.vision import PROFILES, params_for, process

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
WEB = ROOT / "web"
RESULTS = ROOT / "results" / "real_eval.json"   # detector comparison on the D2 TEST years (2018+2021)

# Real side-scan examples from the MILCO dataset (CC BY 4.0, bundled in web/samples/real/ with
# attribution): three where the classical pipeline finds the labelled object and one where it
# does not. If the bundled copy is missing they fall back to the local dataset, and are hidden
# when neither exists. HEXADEPTH_REAL_DATA overrides the dataset location (e.g. to test
# without it).
REAL_DATA = Path(os.environ.get("HEXADEPTH_REAL_DATA", ROOT / "data" / "real" / "milco"))
REAL_WEB = WEB / "samples" / "real"
REAL_SAMPLES = [("2018", "0440_2018.jpg"), ("2010", "0307_2010.jpg"),
                ("2017", "0009_2017.jpg"), ("2021", "0001_2021.jpg")]
REAL_CREDIT = "MILCO dataset, Santos et al. (2024), CC BY 4.0"
REAL_ATTRIBUTION = dict(
    authors="Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C.",
    year=2024,
    title="Side-scan sonar imaging data of underwater vehicles for mine detection",
    doi="https://doi.org/10.6084/m9.figshare.24574879",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Images not resized (already ≤ 1024 px wide); outlines added by HexaDepth.",
)

DISPLAY_WIDTH = 800             # images returned to the browser are scaled to this width
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Same mission defaults as the Streamlit dashboard (app.py).
MISSION = dict(uuv_id=620, drone_x_m=12_450, drone_y_m=30_870, drone_depth_m=18.5)
METERS_PER_PIXEL = synth.METERS_PER_PIXEL   # nominal for real images too (resolution not published)

_SCENE_LABELS = {s.name: s.description for s in synth.default_scenes()}

# Real samples use the best available trained detector (authors' YOLOv4 -> our YOLOv8n ->
# classical); synthetic samples and uploads keep the classical pipeline. Loaded once, on first use.
_real_detector: tuple[object, str | None] | None = None


def real_detector():
    global _real_detector
    if _real_detector is None:
        params = params_for("real", meters_per_pixel=METERS_PER_PIXEL)
        _real_detector = dets.resolve(dets.REAL_SAMPLE_PREFERENCE, params)
    return _real_detector

@lru_cache(maxsize=4)
def _load_sonar_detector(path: str, modified_ns: int):
    """Load once per checkpoint version (including when training replaces the file)."""
    return dets.SonarYoloDetector(
        params_for("real", meters_per_pixel=METERS_PER_PIXEL), weights=Path(path)
    )


def active_sonar_detector():
    """Return (trained detector, note), or (None, reason) without inventing AI results."""
    checkpoint = dets.find_sonar_yolo()
    if checkpoint is None:
        return None, "HexaDepth YOLO checkpoint is not installed locally."
    try:
        return _load_sonar_detector(
            str(checkpoint.resolve()), checkpoint.stat().st_mtime_ns
        ), None
    except (dets.ModelUnavailable, RuntimeError, OSError, ValueError) as exc:
        return None, f"HexaDepth YOLO unavailable: {type(exc).__name__}: {exc}"


app = FastAPI(title="HexaDepth", version="1.1")


# ---------------------------------------------------------------- helpers
def sample_paths() -> list[Path]:
    """Sample images, generated with hexadepth.synth the first time if the folder is empty."""
    if not SAMPLES.exists() or not any(p.suffix.lower() in IMAGE_SUFFIXES for p in SAMPLES.iterdir()):
        synth.write_samples(SAMPLES)
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def real_samples() -> list[dict]:
    """Real examples that are actually available on this machine."""
    out = []
    for year, file in REAL_SAMPLES:
        path = REAL_WEB / file
        if not path.exists():
            path = REAL_DATA / year / year / file
        if path.exists():
            out.append(dict(name=f"real_{year}_{Path(file).stem.split('_')[0]}{Path(file).suffix}",
                            path=path, year=year))
    return out


DRISHTI_WEB = WEB / "samples" / "drishti"
DRISHTI_SAMPLES = [
    ("pipe_1693569376.780_x1500.jpg", "Pipeline · 01"),
    ("pipe_1693569378.780_x1500.jpg", "Pipeline · 02"),
    ("wreckA_DM_Wilson_03_y3520_x1088.jpg", "Shipwreck · 01"),
    ("wreckA_DM_Wilson_03_y3520_x640.jpg", "Shipwreck · 02"),
    ("mine_0001_2021.jpg", "Cylindrical object · 01"),
    ("mine_0002_2015.jpg", "Cylindrical object · 02"),
    ("bg_1693569219.799_x1000.jpg", "Background · no labeled object"),
]


def find_sample(name: str) -> tuple[Path, str]:
    """(path, vision profile) for a listed sample. Exact match only, so no path traversal."""
    for filename, _ in DRISHTI_SAMPLES:
        if name == filename and (DRISHTI_WEB / filename).is_file():
            return DRISHTI_WEB / filename, "default"
    for p in sample_paths():
        if p.name == name:
            return p, "default"
    for r in real_samples():
        if r["name"] == name:
            return r["path"], "real"
    raise HTTPException(404, f"Unknown sample '{name}'")


def sample_short(path: Path) -> str:
    """Short display name: '01_pipeline.png' -> 'Pipeline'."""
    stem = path.stem
    head, _, rest = stem.partition("_")
    key = rest if head.isdigit() and rest else stem
    return key.replace("_", " ").replace("-", " ").title()


def sample_label(path: Path) -> str:
    key = path.stem.split("_", 1)[-1]
    return _SCENE_LABELS.get(key, sample_short(path))


def png_b64(img: np.ndarray, width: int | None = DISPLAY_WIDTH) -> str:
    if width and img.shape[1] > width:
        h = int(round(img.shape[0] * width / img.shape[1]))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Could not encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def detection_dict(det: pk.Detection) -> dict:
    return dict(
        version=det.version,
        uuv_id=det.uuv_id,
        timestamp=det.timestamp,
        object_type=dets.class_name(det.class_id, det.class_name),
        class_id=det.class_id,
        confidence=det.confidence,
        size_m=round(det.extent_m, 2),
        drone_position=dict(x_m=det.drone_x_m, y_m=det.drone_y_m, depth_m=round(det.drone_depth_m, 1)),
        object_offset=dict(dx_m=round(det.obj_dx_m, 1), dy_m=round(det.obj_dy_m, 1)),
        object_position=dict(x_m=round(det.object_x_m, 1), y_m=round(det.object_y_m, 1)),
        vertices=[list(v) for v in det.vertices],
    )


# ---------------------------------------------------------------- API
@app.get("/api/model/status")
def sonar_model_status():
    """Report model installation separately from detection results."""
    checkpoint = dets.find_sonar_yolo()
    return dict(
        installed=checkpoint is not None,
        runtime_installed=importlib.util.find_spec("ultralytics") is not None,
        loaded=_load_sonar_detector.cache_info().currsize > 0,
        model="HexaDepth YOLO11n (DRISHTI)",
        checkpoint_name=dets.SONAR_WEIGHTS,
        classes=[dict(id=20+i, name=dets.class_name(20+i, "Unknown"))
                 for i in range(4)],
        source_dataset="rehan9599/drishti-sss",
        validation_note=("v2 measured 70.5% test precision and 56.4% test recall "
                         "within the DRISHTI dataset; no independent validation. "
                         "Ghost-net PNG examples were excluded from v2 training."),
    )


@app.get("/api/samples")
def list_samples():
    return [dict(
        name=filename, short=label, label=label,
        url=f"/api/samples/{filename}", group="real", profile="default",
        dataset="DRISHTI", split="train", synthetic=False,
        credit="DRISHTI · Rehan Fazal and contributors · CC BY-SA 4.0",
        attribution=dict(
            source="https://huggingface.co/datasets/rehan9599/drishti-sss",
            licence="CC BY-SA 4.0",
            licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
            note="Real training-split image. Sample names describe dataset categories, not model predictions. Not held-out evaluation.",
        ), bundled=True,
    ) for filename, label in DRISHTI_SAMPLES if (DRISHTI_WEB / filename).is_file()]


@app.get("/api/samples/{name}")
def get_sample(name: str):
    return FileResponse(find_sample(name)[0])


@app.get("/api/results")
def results():
    """Measured detector comparison on real sonar, produced by scripts/evaluate_real.py --compare."""
    if not RESULTS.exists():
        raise HTTPException(404, "No results yet. Run scripts/evaluate_real.py --compare")
    out = json.loads(RESULTS.read_text())
    keep = ("detection", "speed", "packets", "size", "recall_upper_bound_pct", "images_with_detection")
    for r in out.get("runs", {}).values():   # drop bulky per-run metadata the page does not use
        m = r["meta"]
        r["meta"] = {k: m.get(k) for k in ("detector", "detector_label", "detector_credit", "training_overlap",
                                           "detector_uses_ground_truth", "detector_settings", "images")}
        for k in list(r):
            if k not in keep and k != "meta":
                del r[k]
    return out


@app.post("/api/analyze")
async def analyze(sample: str | None = Form(None), file: UploadFile | None = File(None),
                  profile: str | None = Form(None), detector_mode: str = Form("auto")):
    if detector_mode not in ("auto", "sonar_yolo", "classical"):
        raise HTTPException(400, "detector_mode must be auto, sonar_yolo, or classical")
    if profile is not None and profile not in PROFILES:
        raise HTTPException(400, f"Unknown profile '{profile}'")
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 25 MB")
        source = file.filename
        profile = profile or "default"
    elif sample:
        path, sample_profile = find_sample(sample)
        data = path.read_bytes()
        source = path.name
        profile = profile or sample_profile
    else:
        raise HTTPException(400, "Send either a 'sample' name or an image 'file'")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Could not read that image file")

    if detector_mode == "classical":
        detector = dets.ClassicalDetector(params_for(profile, meters_per_pixel=METERS_PER_PIXEL))
        note = "Classical OpenCV detector explicitly selected."
    else:
        detector, note = active_sonar_detector()
        if detector is None and detector_mode == "sonar_yolo":
            raise HTTPException(503, note)
        if detector is None:
            use_legacy_real = sample is not None and profile == "real" and (file is None or not file.filename)
            if use_legacy_real:
                detector, legacy_note = real_detector()
                note = "; ".join(filter(None, (note, legacy_note)))
            else:
                detector = dets.ClassicalDetector(params_for(profile, meters_per_pixel=METERS_PER_PIXEL))
                note = (note or "") + " Using classical OpenCV fallback."

    h, w = img.shape
    t0 = time.perf_counter()
    if isinstance(detector, dets.ClassicalDetector):
        res = process(img, detector.params)
        found = [] if not res.found else [dict(
            obj_dx_m=res.obj_dx_m, obj_dy_m=res.obj_dy_m, extent_m=res.extent_m, class_id=res.class_id,
            confidence=res.confidence, vertices=res.grid_vertices, polygon_px=res.polygon_px,
            raw_contour_points=res.raw_contour_points)]
        message, processed, overlay = res.message, res.threshold, res.overlay
    else:
        colour = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        found = detector.detect(colour)
        processed, overlay = box_views(img, found)
        message = (f"{len(found)} object{'s' if len(found) != 1 else ''} detected; the most confident is sent."
                   if found else "No object above the detector's confidence threshold.")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out = dict(
        source=source,
        profile=profile,
        detector=detector.name,
        detector_label=detector.label,
        detector_credit=getattr(detector, "credit", None),
        detector_note=note,
        detections_total=len(found),
        found=bool(found),
        message=message,
        width=w,
        height=h,
        raw_bytes=int(w * h),
        processing_ms=round(elapsed_ms, 1),
        images=dict(
            raw=png_b64(img),
            threshold=png_b64(processed),
            overlay=png_b64(overlay),
        ),
        raw_contour_points=found[0]["raw_contour_points"] if found else 0,
        packet_size=pk.PACKET_SIZE,
    )
    if not found:
        return out

    top = found[0]
    det = pk.Detection(
        uuv_id=MISSION["uuv_id"], timestamp=int(time.time()),
        drone_x_m=MISSION["drone_x_m"], drone_y_m=MISSION["drone_y_m"],
        drone_depth_m=MISSION["drone_depth_m"], **dets.packet_fields(top),
    )
    try:
        hex_str = pk.to_hex(pk.pack(det))
    except pk.PacketError as e:
        raise HTTPException(422, f"Cannot pack detection: {e}") from e

    out.update(
        vertices=[list(v) for v in top["vertices"]],
        polygon_norm=[[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in top["polygon_px"]],
        other_detections=[dict(box_norm=[round(b / d, 5) for b, d in zip(o["box_px"], (w, h, w, h))],
                               object_type=dets.class_name(o["class_id"], ""), confidence=o["confidence"])
                          for o in found[1:] if "box_px" in o],
        object_type=dets.class_name(det.class_id, det.class_name),
        class_id=det.class_id,
        confidence=det.confidence,
        size_m=round(det.extent_m, 2),
        position=dict(
            drone=dict(x_m=det.drone_x_m, y_m=det.drone_y_m, depth_m=det.drone_depth_m),
            object_offset=dict(dx_m=round(det.obj_dx_m, 1), dy_m=round(det.obj_dy_m, 1)),
            object=dict(x_m=round(det.object_x_m, 1), y_m=round(det.object_y_m, 1)),
        ),
        packet_hex=hex_str,
    )
    return out


def box_views(grey: np.ndarray, found: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """'Processed' view (filled outlines of every detection) and overlay for a box detector."""
    processed = np.zeros_like(grey)
    overlay = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(found):
        poly = np.asarray(d["polygon_px"], np.int32)
        cv2.fillPoly(processed, [poly], 255)
        if "box_px" in d:
            x0, y0, x1, y1 = (int(round(v)) for v in d["box_px"])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255) if i else (0, 255, 0), 1)
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 2 if i == 0 else 1)
    return processed, overlay


class TransmitRequest(BaseModel):
    hex: str
    bit_errors: int = Field(0, ge=0, le=64)


@app.post("/api/transmit")
def transmit(req: TransmitRequest):
    try:
        data = pk.from_hex(req.hex)
    except pk.PacketError as e:
        raise HTTPException(400, str(e)) from e
    if not data:
        raise HTTPException(400, "Empty packet")
    n = min(req.bit_errors, len(data) * 8)
    flipped = sorted(random.sample(range(len(data) * 8), n))
    for b in flipped:
        data = pk.flip_bit(data, b)
    return dict(received_hex=pk.to_hex(data), flipped_bits=flipped, bytes=len(data))


class DecodeRequest(BaseModel):
    hex: str


@app.post("/api/decode")
def decode(req: DecodeRequest):
    try:
        det = pk.unpack(pk.from_hex(req.hex))
    except pk.PacketError as e:
        msg = str(e)
        kind = "crc" if "CRC" in msg else "invalid"
        return JSONResponse(status_code=400, content=dict(ok=False, error_type=kind, error=msg))
    return dict(ok=True, integrity="verified", **detection_dict(det))




# ---------------------------------------------------------------- model-independent visual preview
@app.post("/api/preview")
async def visual_preview(sample: str | None = Form(None), file: UploadFile | None = File(None)):
    """Compress ANY grayscale image to 64 bits, without running a detector."""
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 25 MB")
        source = file.filename
    elif sample:
        path, _ = find_sample(sample)
        data = path.read_bytes()
        source = path.name
    else:
        raise HTTPException(400, "Send either a 'sample' or an image 'file'")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(400, "Could not read that image file")
    t0 = time.perf_counter()
    payload = pv.encode_preview(image)
    restored = pv.decode_preview(payload)
    elapsed = (time.perf_counter() - t0) * 1000
    h, w = image.shape
    return dict(
        source=source, width=w, height=h, raw_bytes=int(image.nbytes),
        source_bytes=len(data), payload_bytes=8, payload_bits=64,
        payload_hex=payload.hex().upper(),
        encoding="8×8 one-bit visual preview", ai_required=False, class_prediction=None,
        message="Coarse light/dark map only. No object labels, original details, or navigation metadata.",
        processing_ms=round(elapsed, 2),
        images=dict(raw=png_b64(image), reconstructed=png_b64(pv.enlarge_preview(payload), width=None)),
        preview_grid=restored.tolist(),
    )


class LinkRequest(BaseModel):
    payload_hex: str
    channels: int = Field(1, ge=1, le=4)
    rate_bps: int = Field(100, ge=10, le=20_000)
    corrupt: bool = False
    raw_bytes: int | None = Field(None, ge=0, le=25_000_000)


@app.post("/api/link/simulate")
def link_simulate(req: LinkRequest):
    try:
        payload = pk.from_hex(req.payload_hex)
        return ln.simulate_link(payload, channels=req.channels, rate_bps=req.rate_bps,
                                corrupt=req.corrupt, raw_bytes=req.raw_bytes)
    except (ln.LinkError, pk.PacketError) as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- bit-exact 64-byte frame transport
@app.post("/api/lossless")
async def lossless_transfer(sample: str | None = Form(None), file: UploadFile | None = File(None),
                            channels: int = Form(1), rate_bps: int = Form(2400),
                            corrupt: bool = Form(False)):
    """Application-layer simulation: compress original file, packetize, restore bit-exact bytes."""
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 25 MiB")
        source = file.filename
    elif sample:
        path, _ = find_sample(sample)
        data = path.read_bytes()
        source = path.name
    else:
        raise HTTPException(400, "Choose a sample or upload an image")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(400, "Could not decode the image")
    started = time.perf_counter()
    try:
        outcome = ft.simulate_file_transfer(data, channels=channels, rate_bps=rate_bps,
                                            corrupt=corrupt)
    except ft.TransferError as exc:
        raise HTTPException(400, str(exc)) from exc
    recovered = outcome.pop("restored")
    # Display images are downscaled only for the browser. The actual transferred
    # file bytes are kept untouched and verified by SHA-256 after reassembly.
    preview = None
    if recovered is not None:
        received_img = cv2.imdecode(np.frombuffer(recovered, np.uint8), cv2.IMREAD_UNCHANGED)
        preview = png_b64(received_img)
    return dict(**outcome, source=source, width=int(image.shape[1]),
                height=int(image.shape[0]), ai_required=False,
                processing_ms=round((time.perf_counter() - started) * 1000, 1),
                images={"reconstructed": preview},
                reconstructed_file_base64=(base64.b64encode(recovered).decode("ascii")
                                           if recovered is not None and len(recovered) <= 5 * 1024 * 1024
                                           else None),
                message="Original file bytes reconstructed exactly" if recovered is not None
                else "Corrupt packet rejected; retry needed. No valid image recovered.")


# ---------------------------------------------------------------- priority-first progressive transfer
@app.post("/api/priority")
async def priority_transfer(sample: str | None = Form(None), file: UploadFile | None = File(None),
                            rate_bps: int = Form(2400), corrupt_stage: str | None = Form(None)):
    """Send region coordinates, overview and useful detail before optional exact file.

    The wire-format/CRC/reconstruction are real software tests. Time is modeled;
    no acoustic modem is controlled by this endpoint. A trained detector is used
    only if local weights exist, with untrained saliency as a safe fallback.
    """
    if file is not None and file.filename:
        source_data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(source_data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 25 MiB")
        source = file.filename
    elif sample:
        path, _ = find_sample(sample)
        source_data = path.read_bytes()
        source = path.name
    else:
        raise HTTPException(400, "Choose a sample or upload a sonar image")
    img = cv2.imdecode(np.frombuffer(source_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Could not decode this image")
    detections = None
    active_name = "saliency"
    detector, detector_note = active_sonar_detector()
    if detector is None and (dets.find_yolo() is not None or dets.find_authors_yolov4() is not None):
        try:
            settings = params_for("real", meters_per_pixel=METERS_PER_PIXEL)
            detector = (dets.FineTunedYoloDetector(settings) if dets.find_yolo() is not None
                        else dets.AuthorsYoloV4Detector(settings))
        except (dets.ModelUnavailable, RuntimeError, cv2.error) as exc:
            detector_note = f"Legacy detector unavailable ({type(exc).__name__}); using saliency."
    if detector is not None:
        try:
            detections = detector.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            active_name = detector.name
            detector_note = (f"{detector.label}: unverified on unseen sonar domains; "
                             "saliency fills remaining regions.")
        except (RuntimeError, OSError, ValueError, cv2.error) as exc:
            detector_note = f"Trained inference failed ({type(exc).__name__}); using saliency."
            detections = None
    try:
        result = pr.simulate_progressive(source_data, rate_bps=rate_bps,
                                         detected=detections, corrupt_stage=corrupt_stage)
    except (pr.PriorityError, ft.TransferError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return dict(
        **result, source=source, detector=active_name, detector_note=detector_note,
        detections_total=len(detections or []),
        detected_objects=[
            dict(
                object_type=dets.class_name(d["class_id"], "Unknown"),
                class_id=d["class_id"], confidence=d["confidence"],
                box_px=list(d["box_px"]),
            )
            for d in (detections or []) if "box_px" in d
        ],
    )


# ---------------------------------------------------------------- frontend
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
