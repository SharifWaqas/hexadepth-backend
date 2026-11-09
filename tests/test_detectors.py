"""Batch D2 part 2: pluggable trained detectors, outline-in-box, fallbacks, comparison run."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from hexadepth import detectors as dets
from hexadepth import packet as pk
from hexadepth import realdata as rd
from hexadepth.vision import outline_in_box, params_for

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_real as ev  # noqa: E402

REAL_SAMPLE = ROOT / "web" / "samples" / "real" / "0440_2018.jpg"
HAS_AUTHORS = dets.find_authors_yolov4() is not None and hasattr(cv2.dnn, "readNetFromDarknet")
MISSION = dict(uuv_id=1, timestamp=1, drone_x_m=0, drone_y_m=0, drone_depth_m=0)


def _target_image():
    """Seabed at 100 with a bright highlight and a dark shadow beside it."""
    img = np.full((200, 200), 100, np.uint8)
    cv2.ellipse(img, (90, 100), (12, 8), 0, 0, 360, 230, -1)
    cv2.rectangle(img, (104, 92), (125, 108), 20, -1)
    return img


def _roundtrip(fields):
    return pk.unpack(pk.pack(pk.Detection(**dets.packet_fields(fields), **MISSION)))


# ---------------------------------------------------------------- outline inside a box
def test_outline_in_box_fits_the_box_and_has_shape():
    box = (74.0, 88.0, 128.0, 112.0)
    poly, raw_pts = outline_in_box(_target_image(), box, params_for("real"))
    assert 3 <= len(poly) <= pk.MAX_VERTICES and raw_pts > 4
    assert (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()) == (74, 88, 128, 112)
    assert len(poly) != 4 or cv2.contourArea(poly) < 54 * 24   # a real outline, not just the rectangle


def test_outline_in_box_falls_back_to_the_rectangle():
    flat = np.full((100, 100), 100, np.uint8)
    poly, _ = outline_in_box(flat, (10, 10, 40, 30))
    assert poly.tolist() == [[10, 10], [40, 10], [40, 30], [10, 30]]
    tiny, _ = outline_in_box(flat, (50, 50, 52, 51))
    assert len(tiny) == 4


def test_box_fields_survive_the_64_byte_packet():
    poly, _ = outline_in_box(_target_image(), (74, 88, 128, 112), params_for("real"))
    f = dict(**dets.fields_from_polygon(poly, 200, 200, 0.05), class_id=11, confidence=87)
    d = _roundtrip(f)
    b = rd.polygon_bbox(rd.decoded_polygon_px(d, 0.05, 200, 200))
    assert abs(b.x0 - 74) <= 1.5 and abs(b.y0 - 88) <= 1.5 and abs(b.x1 - 128) <= 1.5 and abs(b.y1 - 112) <= 1.5
    assert dets.class_name(d.class_id, d.class_name) == "NOMBO (non-mine bottom object)"
    assert dets.class_name(2, "Compact") == "Compact"


# ---------------------------------------------------------------- availability + fallback
def test_missing_models_fall_back_with_a_note(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(dets.ModelUnavailable):
        dets.AuthorsYoloV4Detector(params_for("real"))
    with pytest.raises(dets.ModelUnavailable):
        dets.FineTunedYoloDetector(params_for("real"))
    det, note = dets.resolve("authors_yolov4", params_for("real"))
    assert det.name == "classical" and "authors_yolov4" in note and "yolo" in note
    det, note = dets.resolve("classical", params_for("real"))
    assert det.name == "classical" and note is None


def test_model_dir_env_is_used(tmp_path, monkeypatch):
    (tmp_path / "yolov4-custom.cfg").write_text("[net]\nwidth=320\nheight=256\n")
    (tmp_path / dets.AUTHORS_WEIGHTS).write_bytes(b"")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    cfg, weights = dets.find_authors_yolov4()
    assert cfg.parent == tmp_path and weights.name == dets.AUTHORS_WEIGHTS
    assert dets.AuthorsYoloV4Detector._input_size(cfg) == (320, 256)


@pytest.mark.skipif(not HAS_AUTHORS, reason="authors' YOLOv4 weights (or OpenCV Darknet importer) not available")
def test_authors_yolov4_runs_on_cpu():
    det = dets.AuthorsYoloV4Detector(params_for("real"))
    assert (det.input_w, det.input_h) == (512, 512)
    img = cv2.imread(str(REAL_SAMPLE))
    for b in det.boxes(img):
        assert 0 <= b[0] < b[2] <= img.shape[1] and 0 <= b[1] < b[3] <= img.shape[0]
        assert b[4] in (0, 1) and b[5] >= dets.AUTHORS_CONF
    for f in det.detect(img):
        assert f["class_id"] in (10, 11) and len(pk.pack(pk.Detection(**dets.packet_fields(f), **MISSION))) == 64


@pytest.mark.skipif(dets.find_yolo() is None, reason="models/yolo_milco_v8n.pt not available")
def test_fine_tuned_yolo_runs_on_cpu():
    pytest.importorskip("ultralytics")
    threads = cv2.getNumThreads()
    det = dets.FineTunedYoloDetector(params_for("real"))
    assert cv2.getNumThreads() == threads        # importing ultralytics must not change OpenCV threads
    for f in det.detect(cv2.imread(str(REAL_SAMPLE))):
        assert f["class_id"] in (10, 11) and _roundtrip(f).confidence >= 25


# ---------------------------------------------------------------- evaluation with identical rules
def _fake_test_years(root: Path):
    (root / "Training").mkdir(parents=True)
    (root / "Training" / "obj.names.txt").write_text("MILCO\nNOMBO\n")
    d = root / "2018" / "2018"
    d.mkdir(parents=True)
    img = _target_image()
    cv2.imwrite(str(d / "0001_2018.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    (d / "0001_2018.txt").write_text("1 0.505 0.5 0.27 0.12\n")
    cv2.imwrite(str(d / "0002_2018.jpg"), np.full((200, 200, 3), 100, np.uint8))
    (d / "0002_2018.txt").write_text("")


def test_compare_runs_every_available_detector_on_the_same_images(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no_models"))
    _fake_test_years(tmp_path / "milco")
    c = ev.compare(tmp_path / "milco", quiet=True)
    assert set(c["skipped"]) == {"authors_yolov4", "yolo"} and list(c["runs"]) == ["classical", "oracle"]
    assert c["meta"]["split"] == "d2-test" and "may include images the model was trained on" in c["meta"]["held_out_set"]
    o = c["runs"]["oracle"]["detection"]["iou_0.5"]
    assert o["ground_truth"] == 1 and o["true_positives"] == 1
    md = ev.comparison_markdown(c)
    assert "Not run" in md and "Santos et al. (2024), CC BY 4.0" in md


def test_evaluate_falls_back_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no_models"))
    _fake_test_years(tmp_path / "milco")
    r = ev.evaluate(tmp_path / "milco", "d2-test", "real", None, {}, quiet=True, detector="authors_yolov4")
    m = r["meta"]
    assert m["detector"] == "classical" and m["detector_requested"] == "authors_yolov4"
    assert m["detector_fallback_note"] and any(n.startswith("FALLBACK") for n in m["notes"])
    assert "Fallback" in ev.to_markdown(r)
    with pytest.raises(dets.ModelUnavailable):
        ev.evaluate(tmp_path / "milco", "d2-test", "real", None, {}, quiet=True, detector="yolo", fallback=False)


# ---------------------------------------------------------------- website
def test_real_sample_uses_trained_detector_or_falls_back(tmp_path, monkeypatch):
    client = TestClient(server.app)
    real = next(s for s in client.get("/api/samples").json() if s["group"] == "real")
    synth = next(s for s in client.get("/api/samples").json() if s["group"] == "synthetic")

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))          # hide every model
    monkeypatch.setattr(server, "_real_detector", None)
    r = client.post("/api/analyze", data={"sample": real["name"]}).json()
    assert r["detector"] == "classical" and "authors_yolov4" in r["detector_note"]

    s = client.post("/api/analyze", data={"sample": synth["name"]}).json()
    assert s["detector"] == "classical" and s["detector_note"] is None and s["found"]


@pytest.mark.skipif(not HAS_AUTHORS, reason="authors' YOLOv4 weights not available")
def test_real_sample_with_authors_yolov4(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setattr(server, "_real_detector", None)
    client = TestClient(server.app)
    for s in [s for s in client.get("/api/samples").json() if s["group"] == "real"]:
        r = client.post("/api/analyze", data={"sample": s["name"]}).json()
        assert r["detector"] == "authors_yolov4" and "Santos et al. (2024), CC BY 4.0" in r["detector_credit"]
        assert r["detections_total"] == len(r.get("other_detections", [])) + (1 if r["found"] else 0)
        if r["found"]:
            d = client.post("/api/decode", json={"hex": r["packet_hex"]}).json()
            assert d["ok"] and d["object_type"] in ("MILCO (mine-like object)", "NOMBO (non-mine bottom object)")
