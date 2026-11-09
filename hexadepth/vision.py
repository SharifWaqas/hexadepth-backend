"""
Module 1: The Edge Vision Pipeline.

Raw side-scan image  ->  simplified polygon (3-20 vertices) on a 0-255 grid,
plus real-world size, offset from the vehicle, and a heuristic classification.

Steps:
  1. Grayscale load + downscale for speed
  2. Speckle reduction (median + bilateral)
  3. Range-gain normalisation (each column divided by its median brightness)
  4. Brightness relative to seafloor level; nadir water-column strip masked out
  5. Otsu threshold + morphological clean-up
  6. Contours: drop tiny, border-touching and low-contrast blobs, keep the largest
  7. Ramer-Douglas-Peucker (approxPolyDP), epsilon adapted to land in 3-20 vertices
  8. Uniform-scale normalisation onto a 255 x 255 grid (aspect ratio preserved)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .packet import MAX_VERTICES


@dataclass
class VisionParams:
    meters_per_pixel: float = 0.05     # of the ORIGINAL image
    work_width: int = 800              # processing resolution (speed)
    median_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma: float = 60.0
    min_contrast: float = 1.5          # target must be this many times brighter than seafloor
    mask_nadir: bool = True            # ignore the dark water-column strip under the vehicle
    min_area_frac: float = 0.0008      # of image area
    epsilon_frac: float = 0.02         # of perimeter (starting value)
    max_vertices: int = MAX_VERTICES
    # ---- Options for real side-scan (all default to the original synthetic behaviour) ----
    local_bg_ksize: int = 0            # >0: brightness relative to a local median background of
                                       #     this size (work pixels) instead of one global floor
    threshold_ratio: float = 0.0       # >0: fixed threshold on brightness/background instead of Otsu
    morph_ksize: int = 5               # clean-up kernel (open once, close twice)
    max_area_frac: float = 1.0         # drop blobs larger than this fraction of the image
    select: str = "area"               # pick the candidate by "area", "contrast", or "pair"
                                       # (highlight contrast x shadow depth)
    nadir_gap: int = 0                 # bridge bright columns (e.g. a drawn track line) up to this
                                       # width inside the dark nadir strip
    nadir_pad: int = 12                # safety margin (work pixels) either side of the nadir strip
    min_bg_frac: float = 0.0           # with local_bg: ignore pixels whose local background is darker
                                       # than this fraction of the seafloor level (water column, padding)
    shadow_ratio: float = 0.0          # >0: also find acoustic shadows (darker than this x background)
                                       # and report highlight + adjacent shadow as one object
    pair_gap: int = 5                  # max gap (work pixels) between a highlight and its shadow
    min_shadow_frac: float = 0.15      # shadow pixels needed in a highlight+shadow object


# Named parameter profiles. "default" is the synthetic-data tuning and must stay as-is.
# "real" is tuned ONLY on the MILCO 2015 survey (see scripts/evaluate_real.py).
PROFILES: dict[str, dict] = {
    "default": {},
    "real": dict(
        work_width=1024,       # keep full resolution: targets are ~30 px across
        local_bg_ksize=81,     # brightness relative to the local seabed, not one global level
        threshold_ratio=1.4,   # highlight = 1.4x brighter than its surroundings
        morph_ksize=3,         # smaller clean-up kernel so small highlights survive
        min_area_frac=0.0001,  # targets are ~0.1% of the image
        max_area_frac=0.01,    # ignore large bright seabed patches
        nadir_gap=20,          # bridge the drawn track line inside the water column
        min_bg_frac=0.3,       # ignore water column / black padding (dark local background)
        shadow_ratio=0.8,      # acoustic shadow = darker than 0.8x background
        pair_gap=9,            # join a highlight with shadow up to 9 px away
        select="pair",         # rank by highlight contrast x shadow depth
    ),
}


def params_for(profile: str = "default", **overrides) -> VisionParams:
    """VisionParams for a named profile, with optional field overrides."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown vision profile {profile!r}; choose from {sorted(PROFILES)}")
    return VisionParams(**{**PROFILES[profile], **overrides})


@dataclass
class VisionResult:
    found: bool
    message: str
    raw: np.ndarray
    denoised: np.ndarray | None = None
    normalized: np.ndarray | None = None
    threshold: np.ndarray | None = None
    overlay: np.ndarray | None = None
    raw_contour_points: int = 0
    polygon_px: np.ndarray | None = None              # simplified polygon, original-image pixels
    grid_vertices: list[tuple[int, int]] = field(default_factory=list)
    extent_m: float = 0.0
    obj_dx_m: float = 0.0                              # + = east (right of track)
    obj_dy_m: float = 0.0                              # + = north (up the image)
    class_id: int = 0
    confidence: int = 0
    features: dict = field(default_factory=dict)


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def denoise(img: np.ndarray, p: VisionParams) -> np.ndarray:
    out = cv2.medianBlur(img, _odd(p.median_ksize))
    return cv2.bilateralFilter(out, p.bilateral_d, p.bilateral_sigma, p.bilateral_sigma)


def normalize_range_gain(img: np.ndarray) -> np.ndarray:
    """Flatten the brightness falloff with range by dividing each column by its median."""
    f = img.astype(np.float32)
    col = np.median(f, axis=0)
    col = cv2.GaussianBlur(col.reshape(1, -1), (1, 0), 15).ravel()
    floor_level = np.median(col)
    # Columns that are almost black (the nadir water column) stay dark instead of being boosted.
    col = np.maximum(col, 0.25 * floor_level)
    out = f / col[None, :] * 100.0
    return np.clip(out, 0, 255).astype(np.uint8)


def nadir_mask(img: np.ndarray, pad: int = 12, gap: int = 0) -> np.ndarray:
    """Mask (255) of the dark water-column strip under the vehicle plus a safety margin."""
    col = np.median(img.astype(np.float32), axis=0)
    floor_level = np.median(col)
    w = img.shape[1]
    dark = col < 0.35 * floor_level
    if gap > 0:
        # Close short bright runs (a drawn track line) so the strip is one dark run.
        k = np.ones(2 * gap + 1, np.uint8)
        dark = cv2.erode(cv2.dilate(dark.astype(np.uint8).reshape(1, -1), k.reshape(1, -1)),
                         k.reshape(1, -1)).ravel().astype(bool)
    # Only keep the dark run that touches the centre column.
    mask_cols = np.zeros(w, bool)
    c = w // 2
    if dark[c]:
        lo = hi = c
        while lo > 0 and dark[lo - 1]:
            lo -= 1
        while hi < w - 1 and dark[hi + 1]:
            hi += 1
        mask_cols[max(0, lo - pad):min(w, hi + pad + 1)] = True
    m = np.zeros(img.shape, np.uint8)
    m[:, mask_cols] = 255
    return m


def highlight_map(norm: np.ndarray, p: VisionParams) -> np.ndarray:
    """Brightness relative to the seafloor level; only 'brighter than seafloor' survives.

    After range-gain normalisation the seafloor sits at a roughly constant level, so a robust
    global floor estimate works for objects of any size (a local blur would eat large targets).
    """
    valid = norm[norm > 20]
    floor = float(np.median(valid)) if valid.size else 100.0
    ratio = np.clip((norm.astype(np.float32) + 1.0) / (floor + 1.0), 1.0, 4.0)
    return ((ratio - 1.0) / 3.0 * 255).astype(np.uint8)


def simplify(contour: np.ndarray, p: VisionParams) -> np.ndarray:
    """approxPolyDP with epsilon adjusted until the vertex count is in [3, max_vertices]."""
    perimeter = cv2.arcLength(contour, True)
    # For thin objects (pipes, cables) 2% of the perimeter can exceed the object's width and
    # collapse it into a triangle, so epsilon is also capped by the object's thickness.
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    eps = min(p.epsilon_frac * perimeter, 0.25 * max(min(rw, rh), 1.0))
    approx = cv2.approxPolyDP(contour, eps, True)
    for _ in range(40):
        if len(approx) > p.max_vertices:
            eps *= 1.2
        elif len(approx) < 4 and eps > 0.5:
            eps *= 0.7
        else:
            break
        approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 3:  # degenerate; fall back to the rotated bounding box
        approx = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2).astype(np.int32)
    return approx.reshape(-1, 2)


def to_grid(poly_px: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """Shift to bounding-box origin and scale BOTH axes by one factor onto 0-255."""
    x, y, w, h = cv2.boundingRect(poly_px.astype(np.int32))
    longest = max(w, h, 1)
    scale = 255.0 / longest
    grid = [(int(round((px - x) * scale)), int(round((py - y) * scale))) for px, py in poly_px]
    grid = [(min(max(gx, 0), 255), min(max(gy, 0), 255)) for gx, gy in grid]
    return grid, longest


def classify(contour: np.ndarray, contrast: float) -> tuple[int, int, dict]:
    """Rule-based placeholder classifier from simple shape features. Replace with ML later."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    elongation = long_side / short_side
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    rect_fill = area / (rw * rh) if rw * rh > 0 else 0

    features = dict(elongation=round(elongation, 2), solidity=round(solidity, 3),
                    circularity=round(circularity, 3), rect_fill=round(rect_fill, 3),
                    contrast=round(contrast, 2))

    if elongation > 5:
        cls, fit = 1, min(1.0, (elongation - 5) / 5 + 0.6)
    elif circularity > 0.80 and elongation < 1.4:
        cls, fit = 3, min(1.0, (circularity - 0.8) / 0.15 + 0.6)
    elif rect_fill > 0.75 and solidity > 0.9:
        cls, fit = 2, min(1.0, (rect_fill - 0.75) / 0.2 + 0.6)
    elif solidity < 0.92 or circularity < 0.8:
        cls, fit = 4, 0.6
    else:
        cls, fit = 0, 0.3

    strength = float(np.clip((contrast - 1.2) / 1.5, 0, 1))
    confidence = int(round(100 * (0.6 * fit + 0.4 * strength)))
    return cls, max(0, min(100, confidence)), features


def process(img: np.ndarray, params: VisionParams | None = None) -> VisionResult:
    p = params or VisionParams()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    s = p.work_width / W if W > p.work_width else 1.0
    work = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    h, w = work.shape

    den = denoise(work, p)
    norm = normalize_range_gain(den)
    if p.local_bg_ksize > 0:
        # Small, sharp highlights on a textured seabed: compare each pixel with its surroundings.
        bg = cv2.medianBlur(norm, _odd(p.local_bg_ksize)).astype(np.float32)
        ratio_map = (norm.astype(np.float32) + 1.0) / (bg + 1.0)
    else:
        bg, ratio_map = None, None
    hl = highlight_map(norm, p) if ratio_map is None else \
        ((np.clip(ratio_map, 1.0, 4.0) - 1.0) / 3.0 * 255).astype(np.uint8)

    if p.threshold_ratio > 0:
        cut = int(round((min(max(p.threshold_ratio, 1.0), 4.0) - 1.0) / 3.0 * 255))
        th = np.where(hl >= cut, 255, 0).astype(np.uint8)
    else:
        _, th = cv2.threshold(hl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    valid = None
    if ratio_map is not None and p.min_bg_frac > 0:
        lit = norm[norm > 20]
        floor_level = float(np.median(lit)) if lit.size else 100.0
        valid = bg >= p.min_bg_frac * floor_level
        th[~valid] = 0
    if p.morph_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_ksize, p.morph_ksize))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    nadir = nadir_mask(den, pad=p.nadir_pad, gap=p.nadir_gap) > 0 if p.mask_nadir else None
    if nadir is not None:
        th[nadir] = 0

    highlight = None
    if ratio_map is not None and p.shadow_ratio > 0:
        # Side-scan targets = bright highlight + dark acoustic shadow. Join each highlight with
        # nearby shadow pixels and keep only components that contain both.
        sh = (ratio_map < p.shadow_ratio).astype(np.uint8) * 255
        if valid is not None:
            sh[~valid] = 0
        if nadir is not None:
            sh[nadir] = 0
        if p.morph_ksize > 1:
            sh = cv2.morphologyEx(sh, cv2.MORPH_OPEN, k, iterations=1)
        highlight = th
        g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(p.pair_gap), _odd(p.pair_gap)))
        near_hl = cv2.dilate(highlight, g)
        n, lab = cv2.connectedComponents(sh)
        touching = np.unique(lab[(near_hl > 0) & (sh > 0)])
        sh_kept = np.isin(lab, touching[touching > 0]).astype(np.uint8) * 255
        th = cv2.morphologyEx(cv2.bitwise_or(highlight, sh_kept), cv2.MORPH_CLOSE, g, iterations=1)

    result = VisionResult(found=False, message="", raw=img, denoised=den, normalized=norm, threshold=th)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = p.min_area_frac * h * w
    max_area = p.max_area_frac * h * w
    margin = 3
    floor = float(np.median(norm[norm > 20])) if (norm > 20).any() else 1.0
    candidates = []  # (contour, contrast, shadow_depth)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin:
            continue  # touching the image edge: likely partial / not a clean target
        # Otsu ALWAYS splits the image in two, even on empty seafloor. Require the blob to be
        # clearly brighter than the floor so speckle patches are not reported as targets.
        mask = np.zeros_like(norm)
        cv2.drawContours(mask, [c], -1, 255, -1)
        shadow_depth = 1.0
        if ratio_map is None:
            contrast = float(norm[mask > 0].mean()) / max(floor, 1.0)
        elif highlight is not None:
            inside = mask > 0
            hl_px = inside & (highlight > 0)
            n_in = int(inside.sum())
            if not hl_px.any() or n_in == 0:
                continue
            sh_px = inside & (highlight == 0)
            if sh_px.sum() / n_in < p.min_shadow_frac:
                continue  # a highlight without a shadow
            contrast = float(ratio_map[hl_px].mean())
            shadow_depth = 1.0 / max(float(ratio_map[sh_px].mean()), 0.05)
        else:
            contrast = float(ratio_map[mask > 0].mean())
        if contrast < p.min_contrast:
            continue
        candidates.append((c, contrast, shadow_depth))

    overlay = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    if not candidates:
        result.message = "No target found (try another image or adjust parameters)."
        result.overlay = overlay
        return result

    if p.select == "contrast":
        best, contrast, _ = max(candidates, key=lambda t: t[1])
    elif p.select == "pair":
        best, contrast, _ = max(candidates, key=lambda t: t[1] * t[2])
    else:
        best, contrast, _ = max(candidates, key=lambda t: cv2.contourArea(t[0]))
    result.raw_contour_points = len(best)

    poly_work = simplify(best, p)
    poly_px = (poly_work / s).astype(np.int32)  # back to original-image pixels

    grid, longest_px = to_grid(poly_px)
    M = cv2.moments(best)
    cx, cy = M["m10"] / M["m00"] / s, M["m01"] / M["m00"] / s
    cls, conf, feats = classify(best, contrast)

    cv2.drawContours(overlay, [best], -1, (0, 200, 255), 1)
    cv2.polylines(overlay, [poly_work.astype(np.int32)], True, (0, 255, 0), 2)
    for vx, vy in poly_work:
        cv2.circle(overlay, (int(vx), int(vy)), 4, (0, 0, 255), -1)

    result.found = True
    result.message = f"Target found: {len(best)} raw contour points -> {len(grid)} vertices."
    result.overlay = overlay
    result.polygon_px = poly_px
    result.grid_vertices = grid
    result.extent_m = longest_px * p.meters_per_pixel
    # Vehicle assumed at image centre, heading north (up). Simplification for the demo.
    result.obj_dx_m = (cx - W / 2) * p.meters_per_pixel
    result.obj_dy_m = (H / 2 - cy) * p.meters_per_pixel
    result.class_id, result.confidence, result.features = cls, conf, feats
    return result


def outline_in_box(img: np.ndarray, box: tuple[float, float, float, float],
                   params: VisionParams | None = None) -> tuple[np.ndarray, int]:
    """Outline of the object inside a detector box, in original-image pixels.

    Used with box detectors (YOLO): the detector decides WHERE the object is and how big; this
    adds its SHAPE with the same steps as process(): speckle reduction, brightness relative to
    the local seabed, highlight (bright) + acoustic shadow (dark) threshold, morphological
    clean-up, largest contour, approxPolyDP (simplify). The outline is then stretched so its
    bounding box equals the detector box, so the shape never moves or resizes the detection.
    Falls back to the box rectangle when no usable contour is found.

    Returns (polygon_px int32 N x 2, raw contour points).
    """
    p = params or VisionParams()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(W - 1, int(np.ceil(box[2]))), min(H - 1, int(np.ceil(box[3])))
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return rect, 4

    # Local seabed level: median of a window three times the box size around it.
    cx0, cy0 = max(0, x0 - bw), max(0, y0 - bh)
    cx1, cy1 = min(W, x1 + bw + 1), min(H, y1 + bh + 1)
    ctx = cv2.medianBlur(img[cy0:cy1, cx0:cx1], 3).astype(np.float32)
    lit = ctx[ctx > 20]
    bg = float(np.median(lit)) if lit.size else float(np.median(ctx))
    ratio = (ctx + 1.0) / (bg + 1.0)
    hi = p.threshold_ratio if p.threshold_ratio > 0 else 1.4
    lo = p.shadow_ratio if p.shadow_ratio > 0 else 0.8
    mask = ((ratio >= hi) | (ratio < lo)).astype(np.uint8) * 255
    inside = np.zeros_like(mask)
    inside[y0 - cy0:y1 - cy0 + 1, x0 - cx0:x1 - cx0 + 1] = 255
    mask = cv2.bitwise_and(mask, inside)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return rect, 4
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.10 * bw * bh:
        return rect, 4
    poly = simplify(best, p).astype(np.float64)
    if len(poly) < 3:
        return rect, len(best)

    # Fit the outline's bounding box to the detector box.
    mn, mx = poly.min(axis=0), poly.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    fitted = (poly - mn) / span * [bw, bh] + [x0, y0]   # poly is in window coordinates
    fitted = np.round(fitted).astype(np.int32)
    if len(np.unique(fitted, axis=0)) < 3:
        return rect, len(best)
    return fitted, len(best)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    paths = [Path(a) for a in sys.argv[1:]] or sorted(Path("samples").glob("*.png"))
    for path in paths:
        r = process(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        print(f"{path.name:22s} found={r.found} raw_pts={r.raw_contour_points:5d} "
              f"vertices={len(r.grid_vertices):2d} extent={r.extent_m:6.2f} m "
              f"class={r.class_id} conf={r.confidence}")
