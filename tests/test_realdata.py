"""Evaluation maths for the real-data benchmark. Runs without data/real/ (synthetic fixtures)."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from hexadepth import packet as pk
from hexadepth import realdata as rd
from hexadepth.vision import PROFILES, VisionParams, params_for

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HEXADEPTH_REAL_DATA", ROOT / "data" / "real" / "milco"))


# ---------------------------------------------------------------- YOLO labels
def test_parse_yolo_labels_pixels_and_classes():
    boxes = rd.parse_yolo_labels("0 0.5 0.5 0.2 0.1\n1 0.25 0.75 0.1 0.1\n", 1000, 500)
    assert len(boxes) == 2
    b0, b1 = boxes
    assert (b0.cls, b0.x0, b0.y0, b0.x1, b0.y1) == (0, 400, 225, 600, 275)
    assert b1.cls == 1 and b1.centre == pytest.approx((250, 375))


@pytest.mark.parametrize("text", ["", "\n", "   \n\n"])
def test_parse_yolo_labels_empty_file_means_no_objects(text):
    assert rd.parse_yolo_labels(text, 416, 416) == []


def test_parse_yolo_labels_crlf_and_blank_lines():
    assert len(rd.parse_yolo_labels("0 0.5 0.5 0.1 0.1\r\n\r\n1 0.2 0.2 0.1 0.1\r\n", 100, 100)) == 2


def test_parse_yolo_labels_rejects_malformed_line():
    with pytest.raises(ValueError):
        rd.parse_yolo_labels("0 0.5 0.5 0.1\n", 100, 100)


def test_read_class_names_handles_crlf_and_no_final_newline(tmp_path):
    f = tmp_path / "obj.names.txt"
    f.write_bytes(b"MILCO\r\nNOMBO")
    assert rd.read_class_names(f) == ["MILCO", "NOMBO"]


# ---------------------------------------------------------------- IoU / matching / P-R
def test_box_iou_cases():
    a = rd.Box(0, 0, 10, 10)
    assert rd.box_iou(a, a) == 1.0
    assert rd.box_iou(a, rd.Box(20, 20, 30, 30)) == 0.0
    assert rd.box_iou(a, rd.Box(10, 0, 20, 10)) == 0.0                      # touching edges
    assert rd.box_iou(a, rd.Box(5, 0, 15, 10)) == pytest.approx(50 / 150)   # half overlap
    assert rd.box_iou(a, rd.Box(0, 0, 5, 5)) == pytest.approx(25 / 100)     # contained
    assert rd.box_iou(rd.Box(3, 3, 3, 3), rd.Box(3, 3, 3, 3)) == 0.0       # degenerate


def test_match_is_one_to_one_and_respects_threshold():
    gts = [rd.Box(0, 0, 10, 10), rd.Box(100, 100, 110, 110)]
    preds = [rd.Box(1, 1, 11, 11), rd.Box(0, 0, 10, 10), rd.Box(50, 50, 60, 60)]
    m = rd.match(preds, gts, 0.5)
    assert [(i, j) for i, j, _ in m] == [(1, 0)]          # exact box wins; GT used once
    assert rd.match([rd.Box(5, 0, 15, 10)], gts, 0.5) == []  # IoU 0.33 < 0.5
    assert len(rd.match([rd.Box(5, 0, 15, 10)], gts, 0.3)) == 1
    assert rd.match([], gts, 0.3) == [] and rd.match(preds, [], 0.3) == []


def test_precision_recall_and_undefined_cases():
    assert rd.precision_recall(3, 4, 6) == (0.75, 0.5)
    assert rd.precision_recall(0, 0, 5) == (None, 0.0)     # no predictions: precision undefined
    assert rd.precision_recall(0, 2, 0) == (0.0, None)     # no ground truth: recall undefined


# ---------------------------------------------------------------- packet -> pixels
def test_decoded_polygon_round_trips_through_64_bytes():
    W, H, mpp = 1024, 1024, 0.05
    # A 40 x 20 px rectangle centred at (700, 300) in a 1024 x 1024 image.
    grid = [(0, 0), (255, 0), (255, 128), (0, 128)]
    det = pk.Detection(uuv_id=1, timestamp=0, drone_x_m=0, drone_y_m=0, drone_depth_m=0,
                       obj_dx_m=(700 - W / 2) * mpp, obj_dy_m=(H / 2 - 300) * mpp, extent_m=40 * mpp,
                       class_id=2, confidence=90, vertices=grid)
    back = pk.unpack(pk.pack(det))
    box = rd.polygon_bbox(rd.decoded_polygon_px(back, mpp, W, H))
    assert box.centre == pytest.approx((700, 300), abs=2.0)   # decimetre rounding = 2 px at 5 cm/px
    assert box.w == pytest.approx(40, abs=0.5) and box.h == pytest.approx(20, abs=0.5)
    assert rd.decoded_centre_px(back, mpp, W, H) == pytest.approx((700, 300), abs=2.0)


def test_polygon_centroid_square_and_degenerate():
    assert rd.polygon_centroid(np.array([[0, 0], [2, 0], [2, 2], [0, 2]])) == pytest.approx((1, 1))
    assert rd.polygon_centroid(np.array([[0, 0], [2, 0], [4, 0]])) == pytest.approx((2, 0))


# ---------------------------------------------------------------- size comparison
def test_padded_crop_size_and_clamping():
    img = np.zeros((100, 200), np.uint8)
    assert rd.padded_crop(img, rd.Box(50, 40, 70, 60), 0.10).shape == (24, 24)   # 20 + 2 x 2 px
    assert rd.padded_crop(img, rd.Box(0, 0, 10, 10), 0.10).shape == (11, 11)     # clamped at 0
    assert rd.padded_crop(img, rd.Box(195, 95, 200, 100), 0.5).shape == (8, 8)   # clamped at edge
    assert rd.padded_crop(img, rd.Box(30, 30, 30, 30), 0.10).shape == (1, 1)     # degenerate label


def test_jpeg_size_is_real_encoded_size_and_drops_with_quality():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (40, 60), dtype=np.uint8)
    q75, q50 = rd.jpeg_size(crop, 75), rd.jpeg_size(crop, 50)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert q75 == buf.size and q50 < q75
    assert q50 > pk.PACKET_SIZE   # even a tiny noisy crop is far above 64 bytes


def test_send_seconds():
    assert rd.send_seconds(64) == pytest.approx(5.12)
    assert rd.send_seconds(2_000_000) == pytest.approx(160_000)


# ---------------------------------------------------------------- profiles
def test_default_profile_is_the_original_parameters():
    assert PROFILES["default"] == {}
    assert params_for("default") == VisionParams()


def test_real_profile_only_sets_known_fields():
    fields = set(VisionParams.__dataclass_fields__)
    assert set(PROFILES["real"]) <= fields
    assert params_for("real").select == "pair"
    with pytest.raises(ValueError):
        params_for("nope")


# ---------------------------------------------------------------- needs the dataset
@pytest.mark.skipif(not DATA.exists(), reason="data/real/milco not downloaded")
def test_evaluate_real_runs_on_a_few_images(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_real

    report = evaluate_real.evaluate(DATA, "tune", "real", 3, {}, quiet=True)
    assert report["meta"]["images"] == 3
    assert report["packets"]["exactly_64_bytes_pct"] in (None, 100.0)
    assert report["packets"]["crc_valid_pct"] in (None, 100.0)
    assert set(report["detection"]) == {"iou_0.5", "iou_0.3"}


# ---------------------------------------------------------------- box encoding + pluggable detectors
def test_box_to_packet_fields_round_trips_through_64_bytes():
    W, H, mpp = 416, 416, 0.05
    box = rd.Box(300, 100, 340, 120)
    fields = rd.box_to_packet_fields(box, mpp, W, H)
    back = pk.unpack(pk.pack(pk.Detection(uuv_id=1, timestamp=0, drone_x_m=0, drone_y_m=0, drone_depth_m=0,
                                          class_id=0, confidence=100, **fields)))
    out = rd.polygon_bbox(rd.decoded_polygon_px(back, mpp, W, H))
    assert rd.box_iou(out, box) > 0.9


def _fake_dataset(root: Path):
    """Minimal MILCO-shaped dataset: one tune image with a bright target + shadow, one empty."""
    (root / "Training").mkdir(parents=True)
    (root / "Training" / "obj.names.txt").write_bytes(b"MILCO\r\nNOMBO")
    d = root / "2015" / "2015"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(110, 12, (416, 416)), 0, 255).astype(np.uint8)
    img[:, 200:216] = 5                                    # nadir water column
    img[100:110, 300:312] = 250                            # highlight
    img[100:110, 312:330] = 30                             # shadow away from the track
    cv2.imwrite(str(d / "0001_2015.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    (d / "0001_2015.txt").write_text(f"0 {315 / 416} {105 / 416} {30 / 416} {10 / 416}\n")
    cv2.imwrite(str(d / "0002_2015.jpg"), cv2.cvtColor(np.full((416, 416), 110, np.uint8), cv2.COLOR_GRAY2BGR))
    (d / "0002_2015.txt").write_text("")


@pytest.fixture()
def evaluate_real():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_real as ev
    return ev


def test_oracle_detector_reports_encoding_ceiling(tmp_path, evaluate_real):
    _fake_dataset(tmp_path)
    r = evaluate_real.evaluate(tmp_path, "tune", "real", None, {}, quiet=True, detector="oracle")
    d = r["detection"]["iou_0.5"]
    assert r["meta"]["detector"] == "oracle" and r["meta"]["detector_uses_ground_truth"]
    assert (d["true_positives"], d["predictions"], d["ground_truth"]) == (1, 1, 1)
    assert r["meta"]["images_without_objects"] == 1
    assert r["packets"]["exactly_64_bytes_pct"] == 100.0 and r["packets"]["crc_valid_pct"] == 100.0
    assert r["size"]["matched_objects"] == 1 and r["size"]["jpeg_crop_q75"]["median_bytes"] > 64


def test_classical_detector_runs_on_fake_dataset(tmp_path, evaluate_real):
    _fake_dataset(tmp_path)
    r = evaluate_real.evaluate(tmp_path, "tune", "real", None, {}, quiet=True, detector="classical")
    assert r["meta"]["detector"] == "classical" and not r["meta"]["detector_uses_ground_truth"]
    assert r["recall_upper_bound_pct"] == 100.0          # one object, one image with objects
    assert r["meta"]["images"] == 2 and r["speed"]["per_image_ms_median"] is not None


def test_detector_registry_is_pluggable(evaluate_real):
    assert {"classical", "oracle"} <= set(evaluate_real.DETECTORS)
    for cls in evaluate_real.DETECTORS.values():
        assert callable(getattr(cls, "detect")) and hasattr(cls, "uses_ground_truth")
