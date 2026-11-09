import cv2
import numpy as np
import pytest

from hexadepth import packet as pk
from hexadepth import synth, topside
from hexadepth.vision import VisionParams, process

SCENES = synth.default_scenes()


@pytest.mark.parametrize("idx", range(len(SCENES)))
def test_pipeline_finds_target_with_at_most_20_vertices(idx):
    scene = SCENES[idx]
    img = synth.render(scene, seed=idx)
    r = process(img, VisionParams(meters_per_pixel=synth.METERS_PER_PIXEL))
    assert r.found, r.message
    assert 3 <= len(r.grid_vertices) <= 20

    # The detected centroid lands on the true object.
    truth = np.zeros_like(img)
    cv2.fillPoly(truth, [scene.polygon], 255)
    cx = int(round(r.obj_dx_m / synth.METERS_PER_PIXEL + synth.WIDTH / 2))
    cy = int(round(synth.HEIGHT / 2 - r.obj_dy_m / synth.METERS_PER_PIXEL))
    assert truth[cy, cx] == 255 or cv2.pointPolygonTest(scene.polygon, (cx, cy), True) > -15

    # Real-world size within 15% of ground truth.
    _, _, w, h = cv2.boundingRect(scene.polygon)
    true_extent = max(w, h) * synth.METERS_PER_PIXEL
    assert abs(r.extent_m - true_extent) / true_extent < 0.15

    # Classified as the intended class.
    assert r.class_id == scene.class_id


def test_shape_overlap_after_full_round_trip():
    """Vision -> pack -> hex -> unpack -> metres: reconstructed shape matches the truth (IoU)."""
    scene = SCENES[1]  # container
    img = synth.render(scene, seed=1)
    r = process(img, VisionParams(meters_per_pixel=synth.METERS_PER_PIXEL))
    det = pk.Detection(1, 0, 0, 0, 0, r.obj_dx_m, r.obj_dy_m, r.extent_m,
                       r.class_id, r.confidence, r.grid_vertices)
    rx = pk.unpack(pk.from_hex(pk.to_hex(pk.pack(det))))

    pts_m = topside.vertices_in_meters(rx)
    pts_px = pts_m / synth.METERS_PER_PIXEL
    x, y, _, _ = cv2.boundingRect(scene.polygon)
    pts_px = (pts_px + [x, y]).astype(np.int32)  # align bounding-box origins

    a = np.zeros((synth.HEIGHT, synth.WIDTH), np.uint8)
    b = np.zeros_like(a)
    cv2.fillPoly(a, [scene.polygon], 1)
    cv2.fillPoly(b, [pts_px], 1)
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.8, iou


def test_blank_seafloor_reports_no_target():
    rng = np.random.default_rng(0)
    img = (rng.gamma(3, 1 / 3, (600, 800)) * 90).clip(0, 255).astype(np.uint8)
    assert not process(img).found


def test_topside_render_runs():
    det = pk.Detection(620, 1_700_000_000, 100, 200, 18.5, 3.2, -4.1, 12.3, 2, 88,
                       [(0, 0), (255, 0), (255, 100), (0, 100)])
    out = topside.render(det)
    assert out.shape == (720, 720, 3)
