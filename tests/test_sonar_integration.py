"""Integration smoke tests that do not need a GPU or private model checkpoint."""
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

import server
from hexadepth import detectors as dets
from hexadepth import packet as pk


def _image():
    img = np.full((128, 128), 55, dtype=np.uint8)
    cv2.rectangle(img, (16, 20), (82, 75), 185, -1)
    ok, data = cv2.imencode(".png", img)
    assert ok
    return data.tobytes()


class FakeSonarDetector:
    name = "sonar_yolo"
    label = "HexaDepth YOLO11n (test stub)"
    credit = "Mock for endpoint tests"

    def detect(self, img):
        h, w = img.shape[:2]
        poly = np.array([[16, 20], [82, 20], [82, 75], [16, 75]], np.int32)
        return [dict(
            **dets.fields_from_polygon(poly, w, h, 0.05),
            class_id=21, confidence=91, polygon_px=poly,
            box_px=(16., 20., 82., 75.), raw_contour_points=4,
        )]


def test_drishti_model_classes_do_not_collide():
    assert dets.class_name(20, "") == "Submarine pipeline"
    assert dets.class_name(21, "") == "Shipwreck"
    assert "not independently validated" in dets.class_name(22, "")
    assert dets.class_name(23, "") == "Cylindrical underwater object"
    assert dets.class_name(10, "") == "MILCO (mine-like object)"


def test_checkpoint_resolution_explicit_override(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    monkeypatch.setenv("HEXADEPTH_SONAR_WEIGHTS", str(checkpoint))
    assert dets.find_sonar_yolo() is None
    checkpoint.write_bytes(b"placeholder")
    assert dets.find_sonar_yolo() == checkpoint


def test_sonar_predict_box_adapter_without_ultralytics():
    d = dets.SonarYoloDetector.__new__(dets.SonarYoloDetector)
    fake_boxes = SimpleNamespace(
        xyxy=np.array([[2, 3, 26, 35]], dtype=np.float32),
        cls=np.array([1], dtype=np.float32),
        conf=np.array([0.91], dtype=np.float32),
    )
    d.model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=fake_boxes)])
    boxes = d.boxes(np.zeros((48, 48), np.uint8))
    assert len(boxes) == 1
    assert boxes[0][:5] == (2., 3., 26., 35., 1)
    assert abs(boxes[0][5] - 0.91) < 1e-4


def test_model_status_distinguishes_installed_from_loaded(monkeypatch):
    monkeypatch.setattr(dets, "find_sonar_yolo", lambda: None)
    data = TestClient(server.app).get("/api/model/status").json()
    assert data["installed"] is False
    assert data["model"].startswith("HexaDepth")
    assert len(data["classes"]) == 4


def test_analyze_uses_trained_model_and_valid_packet(monkeypatch):
    monkeypatch.setattr(server, "active_sonar_detector", lambda: (FakeSonarDetector(), None))
    client = TestClient(server.app)
    response = client.post("/api/analyze", files={"file": ("sonar.png", _image(), "image/png")},
                           data={"detector_mode": "sonar_yolo"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["detector"] == "sonar_yolo"
    assert data["detections_total"] == 1
    assert data["object_type"] == "Shipwreck"
    assert data["confidence"] == 91
    decoded = pk.unpack(bytes.fromhex(data["packet_hex"]))
    assert decoded.class_id == 21
    assert data["images"]["overlay"].startswith("data:image/png;base64,")


def test_priority_uses_trained_boxes_for_metadata(monkeypatch):
    monkeypatch.setattr(server, "active_sonar_detector", lambda: (FakeSonarDetector(), None))
    response = TestClient(server.app).post(
        "/api/priority",
        files={"file": ("sonar.png", _image(), "image/png")},
        data={"rate_bps": "2400"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ok"] is True
    assert data["model_used"] is True
    assert data["detector"] == "sonar_yolo"
    assert data["detections_total"] == 1
    assert data["detected_objects"][0]["object_type"] == "Shipwreck"
    assert data["regions"][0]["class_id"] == 21
    assert data["simulation_only"] is True


def test_explicit_missing_model_fails_loudly(monkeypatch):
    monkeypatch.setattr(server, "active_sonar_detector",
                        lambda: (None, "Checkpoint missing locally."))
    response = TestClient(server.app).post(
        "/api/analyze", files={"file": ("sonar.png", _image(), "image/png")},
        data={"detector_mode": "sonar_yolo"},
    )
    assert response.status_code == 503
    assert "Checkpoint missing" in response.text
