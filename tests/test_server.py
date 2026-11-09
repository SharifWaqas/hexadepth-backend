import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from hexadepth import packet as pk
from hexadepth import synth


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


@pytest.fixture(scope="module")
def sample_name(client):
    return client.get("/api/samples").json()[0]["name"]


@pytest.fixture(scope="module")
def analysis(client, sample_name):
    r = client.post("/api/analyze", data={"sample": sample_name})
    assert r.status_code == 200, r.text
    return r.json()


def _png_from_data_url(url: str) -> np.ndarray:
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None
    return img


# ---------------------------------------------------------------- frontend
def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Before the full image arrives." in r.text
    for asset in ("/priority.css", "/priority.js", "/style.css", "/app.js"):
        assert client.get(asset).status_code == 200


# ---------------------------------------------------------------- samples
def test_list_samples(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 5
    assert {"name", "short", "label", "url"} <= set(items[0])
    shorts = [i["short"] for i in items]
    assert shorts[:5] == ["Pipeline", "Container", "Tire", "Reef", "Wreck"]
    assert all(len(i["label"]) > len(i["short"]) for i in items[:5])


def test_samples_generated_when_folder_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SAMPLES", tmp_path / "samples")
    names = [s["name"] for s in TestClient(server.app).get("/api/samples").json() if s["group"] == "synthetic"]
    assert len(names) == len(synth.default_scenes())
    assert all((tmp_path / "samples" / n).exists() for n in names)


def test_get_sample_image(client, sample_name):
    r = client.get(f"/api/samples/{sample_name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (synth.HEIGHT, synth.WIDTH)


def test_get_unknown_sample_404(client):
    assert client.get("/api/samples/nope.png").status_code == 404
    assert client.get("/api/samples/..%2Fserver.py").status_code == 404


# ---------------------------------------------------------------- analyze
def test_analyze_sample(analysis):
    a = analysis
    assert a["found"] is True
    assert a["raw_bytes"] == synth.WIDTH * synth.HEIGHT
    for key in ("raw", "threshold", "overlay"):
        assert _png_from_data_url(a["images"][key]).size > 0
    assert 3 <= len(a["vertices"]) <= pk.MAX_VERTICES
    assert all(0 <= x <= 255 and 0 <= y <= 255 for x, y in a["vertices"])
    assert a["object_type"] in pk.CLASS_NAMES.values()
    assert 0 <= a["confidence"] <= 100
    assert a["size_m"] > 0
    assert {"drone", "object_offset", "object"} <= set(a["position"])
    assert len(bytes.fromhex(a["packet_hex"])) == pk.PACKET_SIZE


def test_analyze_upload(client):
    scene = synth.default_scenes()[2]
    ok, buf = cv2.imencode(".png", synth.render(scene, seed=2))
    assert ok
    r = client.post("/api/analyze", files={"file": ("tire.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["found"] and a["source"] == "tire.png"
    assert len(bytes.fromhex(a["packet_hex"])) == 64


def test_analyze_empty_seafloor_reports_not_found(client):
    img = np.full((600, 800), 90, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/analyze", files={"file": ("blank.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    a = r.json()
    assert a["found"] is False and "packet_hex" not in a and a["message"]


def test_analyze_bad_requests(client):
    assert client.post("/api/analyze").status_code == 400
    r = client.post("/api/analyze", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert client.post("/api/analyze", data={"sample": "missing.png"}).status_code == 404


# ---------------------------------------------------------------- transmit
def test_transmit_clean(client, analysis):
    r = client.post("/api/transmit", json={"hex": analysis["packet_hex"], "bit_errors": 0})
    assert r.status_code == 200
    d = r.json()
    assert d["received_hex"] == analysis["packet_hex"]
    assert d["flipped_bits"] == []


@pytest.mark.parametrize("n", [1, 3, 8])
def test_transmit_flips_bits(client, analysis, n):
    d = client.post("/api/transmit", json={"hex": analysis["packet_hex"], "bit_errors": n}).json()
    assert len(d["flipped_bits"]) == n
    a = int(analysis["packet_hex"], 16)
    b = int(d["received_hex"], 16)
    assert bin(a ^ b).count("1") == n


def test_transmit_rejects_bad_input(client):
    assert client.post("/api/transmit", json={"hex": "ZZ", "bit_errors": 0}).status_code == 400
    assert client.post("/api/transmit", json={"hex": "00", "bit_errors": -1}).status_code == 422


# ---------------------------------------------------------------- decode
def test_decode_roundtrip(client, analysis):
    r = client.post("/api/decode", json={"hex": analysis["packet_hex"]})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["integrity"] == "verified"
    assert d["vertices"] == analysis["vertices"]
    assert d["object_type"] == analysis["object_type"]
    assert d["confidence"] == analysis["confidence"]
    assert d["drone_position"]["x_m"] == analysis["position"]["drone"]["x_m"]
    assert abs(d["size_m"] - analysis["size_m"]) < 0.011


def test_decode_corrupted_packet_reports_crc_error(client, analysis):
    rx = client.post("/api/transmit", json={"hex": analysis["packet_hex"], "bit_errors": 1}).json()
    r = client.post("/api/decode", json={"hex": rx["received_hex"]})
    assert r.status_code == 400
    d = r.json()
    assert d["ok"] is False
    assert d["error_type"] == "crc"
    assert "CRC mismatch" in d["error"]


def test_decode_hand_edited_and_malformed(client, analysis):
    h = analysis["packet_hex"]
    edited = ("0" if h[30] != "0" else "1") + h[31:]
    r = client.post("/api/decode", json={"hex": h[:30] + edited})
    assert r.status_code == 400 and r.json()["error_type"] == "crc"

    r = client.post("/api/decode", json={"hex": h[:-2]})
    assert r.status_code == 400 and r.json()["error_type"] == "invalid"
    r = client.post("/api/decode", json={"hex": "not hex"})
    assert r.status_code == 400 and r.json()["ok"] is False


# ---------------------------------------------------------------- real data + results
def test_real_samples_hidden_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REAL_DATA", tmp_path / "nope")
    monkeypatch.setattr(server, "REAL_WEB", tmp_path / "nope2")
    items = TestClient(server.app).get("/api/samples").json()
    assert items and all(i["group"] == "synthetic" for i in items)
    assert TestClient(server.app).get("/api/samples/real_2018_0440.jpg").status_code == 404


@pytest.mark.skipif(not (server.REAL_DATA.exists() or server.REAL_WEB.exists()), reason="real data not available")
def test_real_sample_listed_and_analysed_with_real_profile(client):
    real = [i for i in client.get("/api/samples").json() if i["group"] == "real"]
    assert real and all(i["profile"] == "real" and "MILCO" in i["short"] and i["credit"] for i in real)
    r = client.post("/api/analyze", data={"sample": real[0]["name"]})
    assert r.status_code == 200 and r.json()["profile"] == "real"


def test_analyze_rejects_unknown_profile(client, sample_name):
    assert client.post("/api/analyze", data={"sample": sample_name, "profile": "nope"}).status_code == 400


def test_results_endpoint(tmp_path, monkeypatch):
    f = tmp_path / "real_eval.json"
    f.write_text('{"meta": {"images": 3}, "runs": {"classical": {"meta": {"detector": "classical", "cpu": "x"}, '
                 '"detection": {}, "speed": {}, "debug": 1}}, "headline_detector": "classical"}')
    monkeypatch.setattr(server, "RESULTS", f)
    r = TestClient(server.app).get("/api/results")
    assert r.status_code == 200 and r.json()["meta"]["images"] == 3
    run = r.json()["runs"]["classical"]
    assert "debug" not in run and run["meta"]["detector"] == "classical" and "cpu" not in run["meta"]
    monkeypatch.setattr(server, "RESULTS", tmp_path / "missing.json")
    assert TestClient(server.app).get("/api/results").status_code == 404


def test_committed_results_have_the_fields_the_page_needs(client):
    if not server.RESULTS.exists():
        pytest.skip("results/real_eval.json not generated")
    r = client.get("/api/results").json()
    m = r["meta"]
    assert m["split"] == "d2-test" and m["split_years"] == ["2018", "2021"] and "held_out_set" in m
    assert r["headline_detector"] in r["runs"] and {"classical", "oracle"} <= set(r["runs"])
    for run in r["runs"].values():
        for k in ("iou_0.5", "iou_0.3"):
            d = run["detection"][k]
            assert {"recall_pct", "precision_pct", "false_alarms_per_image", "recall_by_class"} <= set(d)
        assert {"jpeg_crop_q75", "jpeg_crop_q50", "full_image_jpg"} <= set(run["size"])
        assert {"per_image_ms_median", "per_image_ms_p95"} <= set(run["speed"])
        assert run["detection"]["iou_0.5"]["ground_truth"] == m["ground_truth_objects"]   # same images for all
    if "authors_yolov4" in r["runs"]:
        assert "may include images the model was trained on" in r["runs"]["authors_yolov4"]["meta"]["training_overlap"]
