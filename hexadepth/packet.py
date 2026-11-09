"""
Module 2: The Binary Bit-Packer.

Packs one HexaDepth detection into exactly 64 bytes (big-endian).

Layout (offsets in bytes):
    0   B   version            protocol version (currently 1)
    1   H   uuv_id             vehicle ID
    3   I   timestamp          Unix time, seconds
    7   H   drone_x_m          drone easting from local datum, metres (0-65535)
    9   H   drone_y_m          drone northing from local datum, metres (0-65535)
   11   H   drone_depth_dm     drone depth, decimetres (0-6553.5 m)
   13   h   obj_dx_dm          object centroid offset east of drone, decimetres (+/-3276.7 m)
   15   h   obj_dy_dm          object centroid offset north of drone, decimetres
   17   H   extent_cm          real-world length of the object's longest side, centimetres (0-655.35 m)
   19   B   class_id           object classification ID
   20   B   confidence         0-100
   21   B   n_vertices         number of valid (x, y) pairs that follow (3-20)
   22   40B vertices           20 (x, y) pairs on a 0-255 grid, unused pairs zero-filled
   62   H   crc16              CRC-16/CCITT-FALSE over bytes 0-61
                               ----
                               64 bytes total

Vertex coordinates use ONE uniform scale factor for both axes, so the shape
keeps its true proportions. Multiply by extent_cm / 255 to get centimetres.
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1
MAX_VERTICES = 20
PACKET_SIZE = 64

# Body = everything except the CRC.
_BODY_FMT = ">BHIHHHhhHBBB" + f"{MAX_VERTICES * 2}B"
_PACKET_FMT = _BODY_FMT + "H"

assert struct.calcsize(_PACKET_FMT) == PACKET_SIZE, struct.calcsize(_PACKET_FMT)

CLASS_NAMES = {
    0: "Unknown",
    1: "Linear (pipe / cable)",
    2: "Compact (debris / container)",
    3: "Rounded (tire / buoy / drum)",
    4: "Irregular (reef / rock)",
}


class PacketError(ValueError):
    """Raised when a packet cannot be decoded (wrong length, bad CRC, bad fields)."""


@dataclass
class Detection:
    uuv_id: int
    timestamp: int
    drone_x_m: int
    drone_y_m: int
    drone_depth_m: float
    obj_dx_m: float
    obj_dy_m: float
    extent_m: float
    class_id: int
    confidence: int
    vertices: list[tuple[int, int]] = field(default_factory=list)
    version: int = PROTOCOL_VERSION

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"Class {self.class_id}")

    @property
    def object_x_m(self) -> float:
        return self.drone_x_m + self.obj_dx_m

    @property
    def object_y_m(self) -> float:
        return self.drone_y_m + self.obj_dy_m


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    return binascii.crc_hqx(data, 0xFFFF)


def _check_range(name: str, value: int, lo: int, hi: int) -> int:
    if not lo <= value <= hi:
        raise PacketError(f"{name}={value} out of range [{lo}, {hi}]")
    return value


def pack(det: Detection) -> bytes:
    """Encode a Detection into a 64-byte packet."""
    n = len(det.vertices)
    _check_range("n_vertices", n, 3, MAX_VERTICES)

    flat: list[int] = []
    for x, y in det.vertices:
        flat.append(_check_range("vertex x", int(x), 0, 255))
        flat.append(_check_range("vertex y", int(y), 0, 255))
    flat.extend([0] * (MAX_VERTICES * 2 - len(flat)))

    body = struct.pack(
        _BODY_FMT,
        _check_range("version", det.version, 0, 255),
        _check_range("uuv_id", int(det.uuv_id), 0, 0xFFFF),
        _check_range("timestamp", int(det.timestamp), 0, 0xFFFFFFFF),
        _check_range("drone_x_m", int(round(det.drone_x_m)), 0, 0xFFFF),
        _check_range("drone_y_m", int(round(det.drone_y_m)), 0, 0xFFFF),
        _check_range("drone_depth_dm", int(round(det.drone_depth_m * 10)), 0, 0xFFFF),
        _check_range("obj_dx_dm", int(round(det.obj_dx_m * 10)), -32768, 32767),
        _check_range("obj_dy_dm", int(round(det.obj_dy_m * 10)), -32768, 32767),
        _check_range("extent_cm", int(round(det.extent_m * 100)), 0, 0xFFFF),
        _check_range("class_id", int(det.class_id), 0, 255),
        _check_range("confidence", int(det.confidence), 0, 100),
        n,
        *flat,
    )
    packet = body + struct.pack(">H", crc16(body))
    assert len(packet) == PACKET_SIZE
    return packet


def unpack(packet: bytes) -> Detection:
    """Decode a 64-byte packet. Raises PacketError on length or CRC failure."""
    if len(packet) != PACKET_SIZE:
        raise PacketError(f"Packet is {len(packet)} bytes, expected {PACKET_SIZE}")

    fields = struct.unpack(_PACKET_FMT, packet)
    received_crc = fields[-1]
    computed_crc = crc16(packet[:-2])
    if received_crc != computed_crc:
        raise PacketError(
            f"CRC mismatch: packet says 0x{received_crc:04X}, data gives 0x{computed_crc:04X} "
            "(packet corrupted in transit)"
        )

    (version, uuv_id, ts, x_m, y_m, depth_dm, dx_dm, dy_dm, extent_cm,
     class_id, confidence, n) = fields[:12]
    if version != PROTOCOL_VERSION:
        raise PacketError(f"Unsupported protocol version {version}")
    if not 3 <= n <= MAX_VERTICES:
        raise PacketError(f"Invalid vertex count {n}")

    flat = fields[12:12 + MAX_VERTICES * 2]
    vertices = [(flat[2 * i], flat[2 * i + 1]) for i in range(n)]

    return Detection(
        uuv_id=uuv_id,
        timestamp=ts,
        drone_x_m=x_m,
        drone_y_m=y_m,
        drone_depth_m=depth_dm / 10,
        obj_dx_m=dx_dm / 10,
        obj_dy_m=dy_dm / 10,
        extent_m=extent_cm / 100,
        class_id=class_id,
        confidence=confidence,
        vertices=vertices,
        version=version,
    )


def to_hex(packet: bytes) -> str:
    return packet.hex().upper()


def from_hex(hex_string: str) -> bytes:
    cleaned = "".join(hex_string.split())
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise PacketError(f"Not a valid hex string: {exc}") from exc


def flip_bit(packet: bytes, bit_index: int) -> bytes:
    """Simulate acoustic interference by flipping one bit."""
    b = bytearray(packet)
    b[bit_index // 8] ^= 1 << (bit_index % 8)
    return bytes(b)
