"""
Synthetic side-scan sonar generator, used for demos and tests when real data is unavailable.

Each image is 1600 x 1250 grayscale = exactly 2,000,000 bytes of raw pixels.
The vehicle track runs straight up the centre column (dark nadir/water-column strip),
intensity falls off with range, the seafloor has texture, and every object casts an
acoustic shadow pointing away from the track. Speckle is multiplicative (gamma).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1600, 1250  # 2,000,000 pixels
METERS_PER_PIXEL = 0.05     # 80 m swath across the full image


@dataclass
class Scene:
    name: str
    description: str
    polygon: np.ndarray   # ground-truth object outline, pixel coords (N, 2) int32
    class_id: int


def _rotated(points: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return ((points - center) @ rot.T + center).astype(np.int32)


def _rect(cx, cy, w, h, angle):
    pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], float)
    return _rotated(pts, angle, (cx, cy))


def _ellipse(cx, cy, rx, ry, angle, n=60):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)
    return _rotated(pts, angle, (cx, cy))


def _blob(cx, cy, r, rng, n=40, roughness=0.35):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = r * (1 + roughness * (rng.random(n) - 0.5))
    radii = cv2.GaussianBlur(radii.reshape(1, -1).astype(np.float32), (1, 7), 2).ravel()
    pts = np.stack([cx + radii * np.cos(t), cy + radii * np.sin(t)], axis=1)
    return pts.astype(np.int32)


def _hull(cx, cy, length, beam, angle):
    # Simple boat hull: pointed bow, square-ish stern.
    L, B = length / 2, beam / 2
    pts = np.array([
        [cx - L, cy - B], [cx + L * 0.55, cy - B], [cx + L, cy],
        [cx + L * 0.55, cy + B], [cx - L, cy + B],
    ], float)
    return _rotated(pts, angle, (cx, cy))


def default_scenes(seed: int = 7) -> list[Scene]:
    rng = np.random.default_rng(seed)
    return [
        Scene("pipeline", "Exposed pipeline segment (offshore infrastructure)",
              _rect(1150, 620, 520, 34, 25), 1),
        Scene("container", "Lost shipping container (port/channel hazard)",
              _rect(420, 500, 240, 100, -15), 2),
        Scene("tire", "Discarded tire / drum (marine debris)",
              _ellipse(1180, 780, 70, 70, 0), 3),
        Scene("reef", "Oyster reef patch / rock outcrop (habitat)",
              _blob(450, 700, 150, rng), 4),
        Scene("wreck", "Small vessel wreck (post-storm channel survey)",
              _hull(1200, 450, 360, 110, 70), 2),
    ]


def render(scene: Scene, seed: int = 0) -> np.ndarray:
    """Render a scene to a uint8 grayscale side-scan image."""
    rng = np.random.default_rng(seed)
    h, w = HEIGHT, WIDTH
    cx_track = w / 2

    # Seafloor texture: smooth noise at a few scales.
    floor = np.zeros((h, w), np.float32)
    for scale, weight in [(8, 0.5), (32, 0.3), (128, 0.2)]:
        small = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
        floor += weight * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    floor = 0.75 + 0.5 * (floor - floor.min()) / (np.ptp(floor) + 1e-6)

    # Object highlight and shadow masks.
    obj = np.zeros((h, w), np.uint8)
    cv2.fillPoly(obj, [scene.polygon], 255)
    centroid_x = scene.polygon[:, 0].mean()
    direction = 1 if centroid_x > cx_track else -1  # shadow points away from track
    shadow = np.zeros((h, w), np.uint8)
    obj_height_px = 20
    for shift in range(1, obj_height_px * 6, 2):
        m = np.float32([[1, 0, direction * shift], [0, 1, 0]])
        shadow = cv2.bitwise_or(shadow, cv2.warpAffine(obj, m, (w, h)))
    shadow = cv2.bitwise_and(shadow, cv2.bitwise_not(obj))

    refl = floor.copy()
    refl[shadow > 0] *= 0.12
    refl[obj > 0] = 2.6 + 0.4 * rng.random(int((obj > 0).sum()))

    # Range-dependent gain: bright near the track, fading toward the edges.
    across = np.abs(np.arange(w) - cx_track) / (w / 2)
    gain = 1.0 / (1 + 2.2 * across ** 1.5)
    refl *= gain[None, :]

    # Dark water column at nadir.
    nadir_half = 45 + 8 * np.sin(np.arange(h) / 90.0)
    cols = np.abs(np.arange(w)[None, :] - cx_track)
    refl[cols < nadir_half[:, None]] *= 0.05

    # Multiplicative speckle (gamma, mean 1) plus a little additive noise.
    speckle = rng.gamma(shape=3.0, scale=1 / 3.0, size=(h, w)).astype(np.float32)
    img = refl * speckle * 95 + rng.normal(0, 4, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def write_samples(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, scene in enumerate(default_scenes()):
        p = out / f"{i + 1:02d}_{scene.name}.png"
        cv2.imwrite(str(p), render(scene, seed=i))
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "samples"
    for p in write_samples(target):
        print("wrote", p)
