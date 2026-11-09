"""
Helpers for evaluating HexaDepth on real, labelled side-scan sonar (MILCO dataset).

Everything here is plain maths / IO so it can be unit-tested without the dataset:
YOLO label parsing, box IoU, matching, precision/recall, decoding a packet back into
image-pixel boxes, and JPEG crop sizes for the fair size comparison.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import packet as pk

DATASET_NAME = "MILCO side-scan sonar dataset"
DATASET_CITATION = (
    "Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024). "
    "Side-scan sonar imaging data of underwater vehicles for mine detection. "
    "Data in Brief, 53, 110132."
)
DATASET_DOI = "https://doi.org/10.1016/j.dib.2024.110132"
DATASET_DATA_DOI = "https://doi.org/10.6084/m9.figshare.24574879"

# Batch D split (classical detector): tune on one year, test on the other four.
TUNE_YEARS = ("2015",)
TEST_YEARS = ("2010", "2017", "2018", "2021")
# Batch D2 split (trained detector): TEST = whole survey years never used for training or
# validation. TRAIN and VAL share a pool of the other years; VAL is a stratified random slice
# (by survey year x image type) drawn with a fixed seed.
D2_POOL_YEARS = ("2010", "2015", "2017")
D2_TEST_YEARS = ("2018", "2021")
D2_VAL_FRACTION = 0.15
D2_SEED = 0
SPLITS = {
    "tune": TUNE_YEARS,
    "test": TEST_YEARS,
    "d2-train": D2_POOL_YEARS,   # pool minus the VAL slice (see list_split)
    "d2-val": D2_POOL_YEARS,     # the VAL slice only
    "d2-test": D2_TEST_YEARS,
}


def split_scheme(split: str) -> dict:
    """Describe the scheme a split belongs to, for reports: which years were used to fit or
    select parameters, and which were held out for testing."""
    if split.startswith("d2-"):
        return dict(scheme="d2", pool_years=list(D2_POOL_YEARS), val_fraction=D2_VAL_FRACTION,
                    val_seed=D2_SEED, tune_years=list(D2_POOL_YEARS), test_years=list(D2_TEST_YEARS))
    return dict(scheme="d", tune_years=list(TUNE_YEARS), test_years=list(TEST_YEARS))


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in image pixels, (x0, y0) inclusive top-left, (x1, y1) bottom-right."""
    x0: float
    y0: float
    x1: float
    y1: float
    cls: int = -1

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


# ---------------------------------------------------------------- labels
def read_class_names(path: str | Path) -> list[str]:
    """obj.names-style file: one class name per line (handles CRLF and a missing final newline)."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def parse_yolo_labels(text: str, width: int, height: int) -> list[Box]:
    """YOLO lines 'cls xc yc w h' (normalised 0-1) -> pixel boxes. Empty text -> no objects."""
    boxes = []
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"line {line_no}: expected 5 fields, got {len(parts)}: {line!r}")
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:])
        boxes.append(Box((xc - bw / 2) * width, (yc - bh / 2) * height,
                         (xc + bw / 2) * width, (yc + bh / 2) * height, cls))
    return boxes


def _year_pairs(root: Path, years) -> list[tuple[Path, Path]]:
    pairs = []
    for year in years:
        for img in sorted((root / year / year).glob("*.jpg")):
            pairs.append((img, img.with_suffix(".txt")))
    return pairs


def image_stratum(label_text: str) -> str:
    """'nombo' if the image has any NOMBO (class 1), else 'milco' if it has MILCO, else 'empty'."""
    classes = {int(float(ln.split()[0])) for ln in label_text.splitlines() if ln.split()}
    return "nombo" if 1 in classes else "milco" if 0 in classes else "empty"


def d2_val_names(root: str | Path, fraction: float = D2_VAL_FRACTION, seed: int = D2_SEED) -> set[str]:
    """Image file names in the D2 VAL slice: ~fraction of every (survey year, image type) stratum
    of the pool, drawn with a fixed seed. Strata with >= 2 images give at least one image, so VAL
    contains both classes and empty images whenever the pool does."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for img, lbl in _year_pairs(Path(root), D2_POOL_YEARS):
        text = lbl.read_text() if lbl.exists() else ""
        strata[(img.parent.name, image_stratum(text))].append(img.name)
    rng = random.Random(seed)
    val: set[str] = set()
    for key in sorted(strata):
        names = sorted(strata[key])
        k = round(len(names) * fraction)
        if k == 0 and len(names) >= 2:
            k = 1
        val.update(rng.sample(names, k))
    return val


def list_split(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """(image, label) pairs for a split. Layout: <root>/<year>/<year>/*.jpg with matching .txt."""
    root = Path(root)
    pairs = _year_pairs(root, SPLITS[split])
    if split in ("d2-train", "d2-val"):
        val = d2_val_names(root)
        keep_val = split == "d2-val"
        pairs = [(i, l) for i, l in pairs if (i.name in val) == keep_val]
    return pairs


# ---------------------------------------------------------------- geometry / matching
def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match(preds: list[Box], gts: list[Box], thr: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. Returns (pred_idx, gt_idx, iou) with iou >= thr."""
    pairs = sorted(((box_iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr or iou <= 0:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, iou))
    return out


def precision_recall(tp: int, n_pred: int, n_gt: int) -> tuple[float | None, float | None]:
    """None when undefined (no predictions / no ground truth) rather than a flattering 0 or 1."""
    precision = tp / n_pred if n_pred else None
    recall = tp / n_gt if n_gt else None
    return precision, recall


# ---------------------------------------------------------------- packet -> pixels
def polygon_centroid(pts: np.ndarray) -> tuple[float, float]:
    """Area centroid of a simple polygon; falls back to the vertex mean for degenerate shapes."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def decoded_polygon_px(det: pk.Detection, meters_per_pixel: float, width: int, height: int) -> np.ndarray:
    """Rebuild the polygon in image pixels using ONLY decoded packet fields.

    The packet carries the shape on a 0-255 grid (one scale factor, longest side = extent) and the
    object centroid as an offset from the vehicle, which sits at the image centre heading north.
    The shape is placed so its area centroid lands on the decoded centroid.
    """
    grid = np.array(det.vertices, float)
    longest_px = det.extent_m / meters_per_pixel
    rel = grid * (longest_px / 255.0)
    cx = width / 2 + det.obj_dx_m / meters_per_pixel
    cy = height / 2 - det.obj_dy_m / meters_per_pixel
    gx, gy = polygon_centroid(rel)
    return rel + np.array([cx - gx, cy - gy])


def decoded_centre_px(det: pk.Detection, meters_per_pixel: float, width: int, height: int) -> tuple[float, float]:
    return (width / 2 + det.obj_dx_m / meters_per_pixel, height / 2 - det.obj_dy_m / meters_per_pixel)


def polygon_bbox(poly: np.ndarray) -> Box:
    return Box(float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max()))


# ---------------------------------------------------------------- size comparison
def padded_crop(img: np.ndarray, box: Box, pad_frac: float = 0.10) -> np.ndarray:
    """Crop the box grown by pad_frac of its width/height on each side, clamped to the image.
    Always at least 1x1 pixel so degenerate labels still produce a (tiny) crop."""
    h, w = img.shape[:2]
    px, py = box.w * pad_frac, box.h * pad_frac
    x0 = int(np.floor(max(0, box.x0 - px)))
    y0 = int(np.floor(max(0, box.y0 - py)))
    x1 = int(np.ceil(min(w, box.x1 + px)))
    y1 = int(np.ceil(min(h, box.y1 + py)))
    x1, y1 = max(x1, min(w, x0 + 1)), max(y1, min(h, y0 + 1))
    return img[y0:y1, x0:x1]


def jpeg_size(img: np.ndarray, quality: int) -> int:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return int(buf.size)


def send_seconds(n_bytes: float, bps: float = 100.0) -> float:
    return n_bytes * 8 / bps


# ---------------------------------------------------------------- box -> packet fields
def box_to_packet_fields(box: Box, meters_per_pixel: float, width: int, height: int) -> dict:
    """Encode an axis-aligned box (e.g. from a box detector or the ground truth) as packet fields:
    the box outline as a 4-vertex polygon on the 0-255 grid, its centre as an offset from the
    vehicle (image centre, heading north), and its longest side as the extent."""
    longest = max(box.w, box.h)
    scale = 255.0 / longest if longest > 0 else 0.0
    gw, gh = int(round(box.w * scale)), int(round(box.h * scale))
    cx, cy = box.centre
    return dict(
        obj_dx_m=(cx - width / 2) * meters_per_pixel,
        obj_dy_m=(height / 2 - cy) * meters_per_pixel,
        extent_m=longest * meters_per_pixel,
        vertices=[(0, 0), (gw, 0), (gw, gh), (0, gh)],
    )
