"""
Pluggable detectors for real side-scan sonar. Each turns an image into packet fields:

    detect(img_bgr, gts) -> list of dicts with obj_dx_m, obj_dy_m, extent_m, class_id, confidence,
                            vertices (0-255 grid)  [+ optional extras: box_px, polygon_px, ...]

Only detectors with uses_ground_truth = True may read `gts`.

    classical       the OpenCV pipeline in hexadepth/vision.py (one object per image)
    oracle          ground-truth boxes as detections: measures only what the packet preserves
    authors_yolov4  the dataset authors' trained YOLOv4 (Santos et al. 2024, CC BY 4.0), run with
                    OpenCV DNN (Darknet importer, CPU). Needs OpenCV 4.x: 5.0 removed the importer.
    yolo            our YOLOv8n fine-tuned on the Batch D2 train split (Ultralytics, CPU)

Box detectors report every box above the confidence threshold. Each box becomes its own packet:
the outline is extracted inside the box (vision.outline_in_box) and fitted to it.

Model files are NOT in git. Looked up in $MODEL_DIR only when it is set, otherwise in
data/real/milco/Training/ (authors' YOLOv4; $HEXADEPTH_REAL_DATA/Training if set) and models/
(our YOLOv8n). A missing model raises ModelUnavailable; resolve() then falls back to the next
detector with a note instead of crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import realdata as rd
from .vision import outline_in_box, process, to_grid

ROOT = Path(__file__).resolve().parent.parent

# Packet class ids for the dataset's labels. Ids 0-4 are HexaDepth's rule-based shape classes
# (packet.CLASS_NAMES); trained detectors report the dataset classes as 10 + dataset index.
DATASET_CLASS_OFFSET = 10
DATASET_CLASS_NAMES = {10: "MILCO (mine-like object)", 11: "NOMBO (non-mine bottom object)"}

PACKET_KEYS = ("obj_dx_m", "obj_dy_m", "extent_m", "class_id", "confidence", "vertices")

AUTHORS_CFG_NAMES = ("yolov4-custom.cfg", "yolov4-custom.txt")   # the dataset ships the cfg as .txt
AUTHORS_WEIGHTS = "yolov4-custom_5000.weights"
YOLO_WEIGHTS = "yolo_milco_v8n.pt"

# Confidence thresholds are fixed defaults, NOT chosen on test images:
#   authors_yolov4: Darknet's default detection threshold (0.25) and NMS IoU (0.45)
#   yolo:           Ultralytics' default predict settings (conf 0.25, NMS IoU 0.7)
AUTHORS_CONF, AUTHORS_NMS = 0.25, 0.45
YOLO_CONF, YOLO_NMS, YOLO_IMGSZ = 0.25, 0.7, 1024

AUTHORS_CREDIT = ("Detector: YOLOv4 trained by Santos et al. (2024), CC BY 4.0 "
                  "(https://doi.org/10.6084/m9.figshare.24574879)")
AUTHORS_TRAINING_NOTE = (
    "may include images the model was trained on: the authors did not publish which images they "
    "trained and validated on (their notebook builds train.txt / test.txt on Google Drive; the lists "
    "are not in the dataset)")


class ModelUnavailable(RuntimeError):
    """The detector's model file or runtime is missing on this machine."""


def packet_fields(d: dict) -> dict:
    return {k: d[k] for k in PACKET_KEYS}


def model_dirs(kind: str) -> list[Path]:
    """Where to look for model files. $MODEL_DIR, when set, is the only place searched."""
    if os.environ.get("MODEL_DIR"):
        return [Path(os.environ["MODEL_DIR"])]
    if kind == "authors_yolov4":
        data = Path(os.environ.get("HEXADEPTH_REAL_DATA", ROOT / "data" / "real" / "milco"))
        return [data / "Training"]
    return [ROOT / "models"]


def find_authors_yolov4() -> tuple[Path, Path] | None:
    for d in model_dirs("authors_yolov4"):
        weights = d / AUTHORS_WEIGHTS
        for name in AUTHORS_CFG_NAMES:
            if (d / name).is_file() and weights.is_file():
                return d / name, weights
    return None


def find_yolo() -> Path | None:
    for d in model_dirs("yolo"):
        if (d / YOLO_WEIGHTS).is_file():
            return d / YOLO_WEIGHTS
    return None


def fields_from_polygon(poly_px: np.ndarray, width: int, height: int, meters_per_pixel: float) -> dict:
    """Packet fields for a polygon in image pixels. The centroid sent is that of the polygon as the
    receiver will rebuild it from the 0-255 grid, so decoding puts it back in the same place."""
    grid, longest = to_grid(poly_px)
    x, y, _, _ = cv2.boundingRect(poly_px.astype(np.int32))
    gx, gy = rd.polygon_centroid(np.array(grid, float) * (longest / 255.0))
    cx, cy = x + gx, y + gy
    return dict(obj_dx_m=(cx - width / 2) * meters_per_pixel, obj_dy_m=(height / 2 - cy) * meters_per_pixel,
                extent_m=longest * meters_per_pixel, vertices=grid)


# ---------------------------------------------------------------- detectors
class ClassicalDetector:
    """OpenCV pipeline (hexadepth.vision): at most one object per image."""
    name = "classical"
    label = "Classical OpenCV pipeline"
    max_per_image = 1
    uses_ground_truth = False

    def __init__(self, params):
        self.params = params

    def detect(self, img, gts=None):
        res = process(img, self.params)
        if not res.found:
            return []
        return [dict(obj_dx_m=res.obj_dx_m, obj_dy_m=res.obj_dy_m, extent_m=res.extent_m,
                     class_id=res.class_id, confidence=res.confidence, vertices=res.grid_vertices,
                     polygon_px=res.polygon_px, raw_contour_points=res.raw_contour_points)]


class OracleDetector:
    """Perfect detection: every ground-truth box, encoded as a 4-vertex outline. NOT a detector;
    it isolates what the 64-byte encoding preserves (the ceiling for any box detector)."""
    name = "oracle"
    label = "Oracle (ground-truth boxes)"
    max_per_image = None
    uses_ground_truth = True

    def __init__(self, params):
        self.params = params

    def detect(self, img, gts):
        h, w = img.shape[:2]
        return [dict(**rd.box_to_packet_fields(g, self.params.meters_per_pixel, w, h),
                     class_id=DATASET_CLASS_OFFSET + g.cls if g.cls in (0, 1) else 0, confidence=100)
                for g in gts]


class BoxDetector:
    """Shared step for trained box detectors: boxes -> outline inside each box -> packet fields."""
    max_per_image = None
    uses_ground_truth = False

    def __init__(self, params):
        self.params = params

    def boxes(self, img) -> list[tuple[float, float, float, float, int, float]]:
        """(x0, y0, x1, y1, dataset class index, confidence 0-1), image pixels."""
        raise NotImplementedError

    def detect(self, img, gts=None):
        h, w = img.shape[:2]
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        out = []
        for x0, y0, x1, y1, cls, conf in sorted(self.boxes(img), key=lambda b: -b[5]):
            poly, raw_pts = outline_in_box(grey, (x0, y0, x1, y1), self.params)
            out.append(dict(**fields_from_polygon(poly, w, h, self.params.meters_per_pixel),
                            class_id=DATASET_CLASS_OFFSET + int(cls),
                            confidence=int(round(100 * min(max(conf, 0.0), 1.0))),
                            polygon_px=poly, raw_contour_points=raw_pts,
                            box_px=(float(x0), float(y0), float(x1), float(y1))))
        return out


class AuthorsYoloV4Detector(BoxDetector):
    """The dataset authors' trained YOLOv4 (Darknet), run on CPU with OpenCV DNN."""
    name = "authors_yolov4"
    label = "YOLOv4 trained by Santos et al. (2024)"
    credit = AUTHORS_CREDIT

    def __init__(self, params, files: tuple[Path, Path] | None = None):
        super().__init__(params)
        files = files or find_authors_yolov4()
        if files is None:
            where = ", ".join(str(d) for d in model_dirs(self.name))
            raise ModelUnavailable(f"authors' YOLOv4 ({AUTHORS_WEIGHTS} + cfg) not found in {where}")
        if not hasattr(cv2.dnn, "readNetFromDarknet"):
            raise ModelUnavailable(f"OpenCV {cv2.__version__} has no Darknet importer (needs OpenCV 4.x)")
        self.cfg, self.weights = files
        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg), str(self.weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.input_w, self.input_h = self._input_size(self.cfg)

    @staticmethod
    def _input_size(cfg: Path) -> tuple[int, int]:
        size = {}
        for line in cfg.read_text(errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("[") and line != "[net]" and size:
                break
            if "=" in line:
                k, v = (s.strip() for s in line.split("=", 1))
                if k in ("width", "height"):
                    size[k] = int(v)
        return size.get("width", 416), size.get("height", 416)

    def boxes(self, img):
        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Darknet: plain resize to the network size (no letterbox), RGB, 0-1.
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.input_w, self.input_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        rows = np.concatenate([np.asarray(o).reshape(-1, np.asarray(o).shape[-1]) for o in self.net.forward(self.out_names)])
        scores = rows[:, 5:]                      # objectness x class probability, as in Darknet
        cls = scores.argmax(1)
        conf = scores[np.arange(len(rows)), cls]
        keep = conf >= AUTHORS_CONF
        rows, cls, conf = rows[keep], cls[keep], conf[keep]
        if not len(rows):
            return []
        xywh = [[float((r[0] - r[2] / 2) * w), float((r[1] - r[3] / 2) * h), float(r[2] * w), float(r[3] * h)] for r in rows]
        idx = cv2.dnn.NMSBoxesBatched(xywh, conf.astype(float).tolist(), cls.astype(int).tolist(),
                                      AUTHORS_CONF, AUTHORS_NMS)
        out = []
        for i in np.asarray(idx).reshape(-1):
            x, y, bw, bh = xywh[i]
            out.append((max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh), int(cls[i]), float(conf[i])))
        return out


class FineTunedYoloDetector(BoxDetector):
    """Our YOLOv8n, fine-tuned on the Batch D2 train split (2010+2015+2017 minus the val slice)."""
    name = "yolo"
    label = "YOLOv8n fine-tuned by HexaDepth (D2 split)"

    def __init__(self, params, weights: Path | None = None):
        super().__init__(params)
        weights = weights or find_yolo()
        if weights is None:
            where = ", ".join(str(d) for d in model_dirs(self.name))
            raise ModelUnavailable(f"{YOLO_WEIGHTS} not found in {where}")
        threads = cv2.getNumThreads()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelUnavailable("ultralytics is not installed (pip install -r requirements-yolo.txt)") from e
        finally:
            cv2.setNumThreads(threads)   # importing ultralytics sets OpenCV to 1 thread; keep timings fair
        self.weights = weights
        self.model = YOLO(str(weights))

    def boxes(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        r = self.model.predict(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_NMS, device="cpu", verbose=False)[0]
        b = r.boxes
        return [(float(x0), float(y0), float(x1), float(y1), int(c), float(s))
                for (x0, y0, x1, y1), c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist())]


DETECTORS = {
    "authors_yolov4": AuthorsYoloV4Detector,
    "yolo": FineTunedYoloDetector,
    "classical": ClassicalDetector,
    "oracle": OracleDetector,
}
# When a model is missing, use the next one that is available (oracle is never a fallback).
FALLBACKS = {"authors_yolov4": ("yolo", "classical"), "yolo": ("classical",)}
# Preference for the website's real samples (best measured first; see results/real_eval.md).
REAL_SAMPLE_PREFERENCE = "authors_yolov4"


def resolve(name: str, params) -> tuple[object, str | None]:
    """Instantiate `name`, or the first available fallback. Returns (detector, note or None)."""
    reasons = []
    for candidate in (name, *FALLBACKS.get(name, ())):
        try:
            det = DETECTORS[candidate](params)
        except ModelUnavailable as e:
            reasons.append(f"{candidate}: {e}")
            continue
        note = None if candidate == name else f"Using '{candidate}' instead of '{name}' ({'; '.join(reasons)})."
        return det, note
    raise ModelUnavailable("; ".join(reasons))


def class_name(class_id: int, default: str) -> str:
    return DATASET_CLASS_NAMES.get(class_id, default)
