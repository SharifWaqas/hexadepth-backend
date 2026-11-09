"""Model-independent, deliberately coarse 64-bit visual preview.

A 64-bit payload can carry an 8x8 binary map, NOT a meaningful full image or
reliable object classification. The receiver can show occupancy/brightness
structure; no neural-network weights or training dataset are needed.

Framing, navigation metadata and error detection are NOT included in these eight
payload bytes. The transport layer adds those bytes on the wire.
"""
from __future__ import annotations

import cv2
import numpy as np

PREVIEW_SIDE = 8
PREVIEW_BYTES = 8


def encode_preview(image: np.ndarray) -> bytes:
    """Downsample any 2D uint8 grayscale image to an 8x8 one-bit intensity map."""
    if image is None or image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Expected a non-empty 8-bit grayscale image")
    # INTER_AREA averages large regions rather than picking one pixel; works for
    # arbitrary image sizes. No object detector, learned classes, or model weights.
    small = cv2.resize(image, (PREVIEW_SIDE, PREVIEW_SIDE), interpolation=cv2.INTER_AREA)
    p10, p90 = np.percentile(small, [10, 90])
    if p90 - p10 < 8:
        bits = np.zeros((PREVIEW_SIDE, PREVIEW_SIDE), dtype=np.uint8)
    else:
        midpoint = (float(p10) + float(p90)) / 2.0
        bits = (small >= midpoint).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="big").tobytes()
    assert len(packed) == PREVIEW_BYTES
    return packed


def decode_preview(payload: bytes) -> np.ndarray:
    """Return 8x8 monochrome image from EXACTLY 64 bits (no CRC in payload)."""
    if len(payload) != PREVIEW_BYTES:
        raise ValueError(f"Visual preview needs {PREVIEW_BYTES} bytes, got {len(payload)}")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    return bits.reshape((PREVIEW_SIDE, PREVIEW_SIDE)).astype(np.uint8) * 255


def enlarge_preview(payload: bytes, pixel_size: int = 32) -> np.ndarray:
    """Blown-up block grid for display; creates no new information."""
    return cv2.resize(decode_preview(payload),
                      (PREVIEW_SIDE * pixel_size, PREVIEW_SIDE * pixel_size),
                      interpolation=cv2.INTER_NEAREST)
