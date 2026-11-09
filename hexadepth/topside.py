"""
Topside reconstruction: draws a decoded Detection on a simulated sonar screen.
Uses ONLY what is inside the 64-byte packet.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from .packet import Detection

SCREEN_BG = (60, 25, 8)        # dark navy (BGR)
GRID = (110, 70, 30)
TARGET_FILL = (90, 230, 120)   # green fill
TARGET_EDGE = (200, 255, 210)
TEXT = (235, 235, 235)
TEXT_DIM = (200, 180, 150)


def vertices_in_meters(det: Detection) -> np.ndarray:
    """Grid (0-255) -> metres, using the single uniform scale carried in extent_m."""
    return np.array(det.vertices, float) * (det.extent_m / 255.0)


def render(det: Detection, size: int = 720) -> np.ndarray:
    """Return a BGR image of the reconstructed target."""
    img = np.full((size, size, 3), SCREEN_BG, np.uint8)
    c = size // 2

    # Sonar-style range rings and crosshair.
    for r in range(size // 8, size, size // 8):
        cv2.circle(img, (c, c), r, GRID, 1, cv2.LINE_AA)
    cv2.line(img, (c, 0), (c, size), GRID, 1)
    cv2.line(img, (0, c), (size, c), GRID, 1)

    # Fit the shape into the middle 60% of the screen.
    pts_m = vertices_in_meters(det)
    span = max(np.ptp(pts_m[:, 0]), np.ptp(pts_m[:, 1]), 1e-6)
    px_per_m = (size * 0.6) / span
    centre_m = (pts_m.max(axis=0) + pts_m.min(axis=0)) / 2
    pts_px = ((pts_m - centre_m) * px_per_m + c).astype(np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_px], TARGET_FILL, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    cv2.polylines(img, [pts_px], True, TARGET_EDGE, 2, cv2.LINE_AA)
    for x, y in pts_px:
        cv2.circle(img, (int(x), int(y)), 4, TARGET_EDGE, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, det.class_name, (16, 36), font, 0.9, TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, f"Confidence: {det.confidence}%", (16, 70), font, 0.75, TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, f"Class ID: {det.class_id}", (16, 100), font, 0.75, TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, f"Size: {det.extent_m:.2f} m (longest side)", (16, 130), font, 0.75, TEXT, 2, cv2.LINE_AA)

    when = datetime.fromtimestamp(det.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"UUV {det.uuv_id}  |  {when}",
        f"Target  E {det.object_x_m:.1f} m  N {det.object_y_m:.1f} m",
        f"Vehicle depth {det.drone_depth_m:.1f} m",
    ]
    for i, line in enumerate(reversed(lines)):
        cv2.putText(img, line, (16, size - 18 - 28 * i), font, 0.6, TEXT_DIM, 1, cv2.LINE_AA)

    # Scale bar: a round number of metres close to 1/5 of the screen.
    target_m = size / 5 / px_per_m
    nice = min([0.5, 1, 2, 5, 10, 20, 50, 100, 200], key=lambda v: abs(v - target_m))
    bar_px = int(nice * px_per_m)
    x0, y0 = size - bar_px - 24, 44
    cv2.line(img, (x0, y0), (x0 + bar_px, y0), TEXT, 2)
    cv2.line(img, (x0, y0 - 6), (x0, y0 + 6), TEXT, 2)
    cv2.line(img, (x0 + bar_px, y0 - 6), (x0 + bar_px, y0 + 6), TEXT, 2)
    cv2.putText(img, f"{nice:g} m", (x0, y0 - 12), font, 0.65, TEXT, 2, cv2.LINE_AA)
    return img
