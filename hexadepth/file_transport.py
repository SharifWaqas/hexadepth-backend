"""Bit-exact file transfer over 64-byte framed packets (acoustic-link *simulation*).

Codec uses zlib only if it saves bytes, otherwise transmits source bytes as-is.
Every packet is exactly 64 bytes including an 18-byte header/CRC and at most 46
bytes of payload. A SHA-256 digest verifies the *original file bytes* after
reassembly. This is an application-layer protocol, not an acoustic modem or FEC.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass

from hexadepth.packet import crc16

FRAME_BYTES = 64
HEADER_BYTES = 16
CRC_BYTES = 2
PAYLOAD_BYTES = FRAME_BYTES - HEADER_BYTES - CRC_BYTES  # 46 bytes
FILE_HEADER_BYTES = 4 + 1 + 4 + 32
MAX_FILE_BYTES = 25 * 1024 * 1024
MAGIC = b"HD"
FILE_MAGIC = b"HXD1"
VERSION = 1
FRAME_STRUCT = struct.Struct(">2sBIIIB")

class TransferError(ValueError):
    """Packet, file, or integrity check failed."""


def encode_file(data: bytes) -> tuple[bytes, str]:
    if not 1 <= len(data) <= MAX_FILE_BYTES:
        raise TransferError("File must contain 1 to 25 MiB")
    zipped = zlib.compress(data, level=6)
    use_zip = len(zipped) < len(data)
    header = FILE_MAGIC + struct.pack(">BI", 1 if use_zip else 0, len(data)) + hashlib.sha256(data).digest()
    return header + (zipped if use_zip else data), "zlib" if use_zip else "stored"


def decode_file(payload: bytes) -> bytes:
    if len(payload) < FILE_HEADER_BYTES or payload[:4] != FILE_MAGIC:
        raise TransferError("Missing file envelope")
    flag, expected_size = struct.unpack(">BI", payload[4:9])
    if flag not in (0, 1) or not 1 <= expected_size <= MAX_FILE_BYTES:
        raise TransferError("Invalid envelope")
    content = payload[FILE_HEADER_BYTES:]
    if flag:
        try:
            d = zlib.decompressobj()
            decoded = d.decompress(content, expected_size + 1)
            decoded += d.flush()
            if not d.eof or d.unused_data or d.unconsumed_tail:
                raise TransferError("Incomplete or trailing compressed data")
        except zlib.error as exc:
            raise TransferError("Invalid compressed data") from exc
    else:
        decoded = content
    if len(decoded) != expected_size or hashlib.sha256(decoded).digest() != payload[9:41]:
        raise TransferError("File SHA-256/length mismatch: reconstructed bytes are not identical")
    return decoded


def frame_file(payload: bytes, transfer_id: int | None = None) -> list[bytes]:
    if not payload:
        raise TransferError("Cannot packetize empty payload")
    if transfer_id is None:
        transfer_id = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    total = math.ceil(len(payload) / PAYLOAD_BYTES)
    frames = []
    for index in range(total):
        chunk = payload[index * PAYLOAD_BYTES:(index + 1) * PAYLOAD_BYTES]
        header = FRAME_STRUCT.pack(MAGIC, VERSION, transfer_id, index, total, len(chunk))
        body = header + chunk.ljust(PAYLOAD_BYTES, b"\x00")
        frames.append(body + crc16(body).to_bytes(2, "big"))
    return frames


def reassemble_file(frames: list[bytes]) -> bytes:
    if not frames:
        raise TransferError("No packets received")
    chunks = {}
    total_expected = session_id = None
    for frame in frames:
        if len(frame) != FRAME_BYTES:
            raise TransferError("Packet must be exactly 64 bytes")
        if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "big"):
            raise TransferError("CRC mismatch: corrupt packet must be retried")
        magic, version, sid, index, count, n = FRAME_STRUCT.unpack(frame[:HEADER_BYTES])
        if magic != MAGIC or version != VERSION or count < 1 or index >= count or n > PAYLOAD_BYTES or n < 1:
            raise TransferError("Invalid packet metadata")
        if total_expected is None:
            total_expected, session_id = count, sid
        if count != total_expected or sid != session_id or index in chunks:
            raise TransferError("Mixed sessions or duplicate packets")
        if any(frame[HEADER_BYTES + n:FRAME_BYTES - CRC_BYTES]):
            raise TransferError("Invalid nonzero packet padding")
        chunks[index] = frame[HEADER_BYTES:HEADER_BYTES + n]
    if len(chunks) != total_expected:
        raise TransferError("Missing packets: cannot reconstruct file; request retransmission")
    return b"".join(chunks[index] for index in range(total_expected))


def simulate_file_transfer(source: bytes, channels: int = 1, rate_bps: int = 2400,
                           corrupt: bool = False) -> dict:
    if channels not in (1, 2, 4):
        raise TransferError("Choose 1, 2, or 4 independent links")
    if not 10 <= rate_bps <= 62_500:
        raise TransferError("Rate must be 10–62,500 bits per second per independent link")
    envelope, method = encode_file(source)
    frames = frame_file(envelope)
    used = min(channels, len(frames))
    wire_bytes = len(frames) * FRAME_BYTES
    lane_stats = []
    for lane in range(used):
        count = (len(frames) - 1 - lane) // used + 1
        b = count * FRAME_BYTES
        lane_stats.append({"lane": lane + 1, "packet_count": count, "wire_bytes": b,
                           "airtime_seconds": round(b * 8 / rate_bps, 3)})
    # Header and setup constants are illustrative. No actual modems, spectrum
    # assignments, error-correction, contention, ACKs or retransmits are modeled.
    setup_and_propagation = 0.48
    single_seconds = wire_bytes * 8 / rate_bps + setup_and_propagation
    parallel_seconds = max(s["wire_bytes"] for s in lane_stats) * 8 / rate_bps + setup_and_propagation
    received = list(frames)
    if corrupt:
        i = len(received) // 2
        bad = bytearray(received[i]); bad[HEADER_BYTES] ^= 1
        received[i] = bytes(bad)
    try:
        restored = decode_file(reassemble_file(received))
        if restored != source:
            raise TransferError("Recovered data mismatches source")
        ok, error = True, None
    except TransferError as exc:
        restored, ok, error = None, False, str(exc)
    return {
        "ok": ok, "error": error, "restored": restored,
        "source_bytes": len(source), "compressed_bytes": len(envelope) - FILE_HEADER_BYTES,
        "encoded_bytes": len(envelope), "codec": method,
        "sha256_source": hashlib.sha256(source).hexdigest(),
        "sha256_received": hashlib.sha256(restored).hexdigest() if restored is not None else None,
        "packet_bytes": FRAME_BYTES, "packet_header_and_crc_bytes": FRAME_BYTES - PAYLOAD_BYTES,
        "payload_bytes_per_packet": PAYLOAD_BYTES, "packet_count": len(frames),
        "wire_bytes": wire_bytes, "first_packet_hex": frames[0].hex().upper(),
        "requested_channels": channels, "channels_used": used, "rate_bps_per_channel": rate_bps,
        "single_link_seconds": round(single_seconds, 3), "parallel_seconds": round(parallel_seconds, 3),
        "estimated_speedup": round(single_seconds / parallel_seconds, 2),
        "fragments": lane_stats,
        "integrity": "CRC-16 per packet + SHA-256 full file verified" if ok else "Rejected: corrupt or incomplete transfer",
        "simulation_only": True,
        "limitations": "Independent modems/bandwidth assumed. Does not model acoustic interference, modem protocol, FEC or retransmissions.",
    }
