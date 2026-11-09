"""Deterministic simulator for packetized underwater acoustic communication.

This is NOT a physical modem or a validated channel model. `channels` represents
INDEPENDENT simultaneous acoustic links, each of which can carry `rate_bps`.
A single modem split into software threads would not have this throughput.
Each fragment includes a version/index/count header and CRC-16; this costs 5
bytes per fragment on the wire (in addition to modem-specific preambles/FEC).
"""
from __future__ import annotations

from dataclasses import dataclass
from hexadepth.packet import crc16

FRAGMENT_OVERHEAD_BYTES = 5
MAX_CHANNELS = 4
VERSION = 1


class LinkError(ValueError):
    pass


def frame_fragments(payload: bytes, channels: int) -> list[bytes]:
    if not 1 <= channels <= MAX_CHANNELS:
        raise LinkError("Number of independent channels must be 1 through 4")
    if not 1 <= len(payload) <= 4096:
        raise LinkError("Payload must contain 1 to 4096 bytes")
    n = min(channels, len(payload))
    # Divide evenly, with larger fragments assigned to the first lanes.
    width, remainder = divmod(len(payload), n)
    chunks = []
    position = 0
    for i in range(n):
        count = width + (1 if i < remainder else 0)
        chunks.append(payload[position:position + count])
        position += count
    framed = []
    for i, part in enumerate(chunks):
        body = bytes([VERSION, i, len(chunks)]) + part
        framed.append(body + crc16(body).to_bytes(2, "big"))
    return framed


def reassemble_fragments(frames: list[bytes]) -> bytes:
    if not frames:
        raise LinkError("No fragments received")
    fragments: dict[int, bytes] = {}
    total = None
    for frame in frames:
        if len(frame) < FRAGMENT_OVERHEAD_BYTES + 1:
            raise LinkError("Fragment too short")
        if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "big"):
            raise LinkError("CRC mismatch: a fragment was corrupted")
        version, index, count = frame[:3]
        if version != VERSION or count < 1 or index >= count:
            raise LinkError("Invalid fragment header")
        if total is None:
            total = count
        if count != total or index in fragments:
            raise LinkError("Inconsistent or duplicate fragment")
        fragments[index] = frame[3:-2]
    if len(fragments) != total:
        raise LinkError("Missing fragments; request a retry")
    return b"".join(fragments[i] for i in range(total))


def simulate_link(payload: bytes, channels: int = 1, rate_bps: int = 100,
                  corrupt: bool = False, raw_bytes: int | None = None) -> dict:
    if not 10 <= rate_bps <= 20_000:
        raise LinkError("Bit rate must be between 10 and 20,000 bps per channel")
    frames = frame_fragments(payload, channels)
    wire_bytes = sum(map(len, frames))
    # Parallel lanes start simultaneously. These constants are clearly marked
    # illustrative, not field-calibrated acoustic hardware data.
    setup_s = 0.15
    propagation_s = 0.33
    one_link_s = (len(payload) + FRAGMENT_OVERHEAD_BYTES) * 8 / rate_bps + setup_s + propagation_s
    parallel_s = max(len(f) for f in frames) * 8 / rate_bps + setup_s + propagation_s
    received = list(frames)
    if corrupt:
        b = bytearray(received[0]); b[3] ^= 0x01; received[0] = bytes(b)
    try:
        restored = reassemble_fragments(received)
        ok = restored == payload
        error = None if ok else "Decoded content mismatch"
    except LinkError as exc:
        ok, restored, error = False, b"", str(exc)
    return {
        "ok": ok,
        "error": error,
        "payload_hex": restored.hex().upper() if ok else None,
        "payload_bytes": len(payload),
        "wire_bytes": wire_bytes,
        "requested_channels": channels,
        "channels_used": len(frames),
        "rate_bps_per_channel": rate_bps,
        "single_link_seconds": round(one_link_s, 3),
        "parallel_seconds": round(parallel_s, 3),
        "raw_link_seconds": round(raw_bytes * 8 / rate_bps + setup_s + propagation_s, 3) if raw_bytes is not None else None,
        "estimated_speedup": round(one_link_s / parallel_s, 2),
        "setup_seconds_assumed": setup_s,
        "propagation_seconds_assumed": propagation_s,
        "hardware_required": "One independent acoustic channel/modem per concurrent lane; not ordinary software threads",
        "fragments": [
            {"lane": i + 1, "payload_bytes": len(f) - FRAGMENT_OVERHEAD_BYTES,
             "wire_bytes": len(f), "wire_hex": f.hex().upper(),
             "airtime_seconds": round(len(f) * 8 / rate_bps, 3)}
            for i, f in enumerate(frames)
        ],
        "integrity": "CRC-16 verified" if ok else "Rejected: corrupted or incomplete",
        "simulation_only": True,
    }
