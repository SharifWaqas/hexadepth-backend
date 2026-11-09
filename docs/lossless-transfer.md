# HexaDepth: lossless file transfer over 64-byte application frames

## What this version really does

1. Acquire a source sonar image **file** (JPEG, PNG, TIFF etc.) without modifying its original bytes. A physical sonar sensor creates the image; a compatible acoustic **data modem** would be needed to transmit it.
2. Try zlib on the original file. If zlib doesn't save bytes, store the file unchanged instead. Already compressed images (e.g., JPEG) often see almost no further reduction. This is **lossless** in both modes.
3. Add a small file envelope: magic, codec flag, source length and a SHA-256 digest. The SHA-256 verification at the receiver proves the original source file was recovered byte-for-byte in this simulation.
4. Split the envelope into frames that are **exactly 64 bytes each**. Each frame has a 16-byte header with magic, version, session id, zero-based packet sequence, total packet count and data length, plus a 2-byte CRC-16. Up to **46 payload bytes** are available. Final-frame unused space is zero-padded.
5. Assign packets round-robin to 1, 2 or 4 independent *simulated* links. Each link is assumed to have its own full configured bit rate (a strong assumption). Receive frames in any order; verify CRC, sequence, completeness, decompress and verify SHA-256.
6. Only show a reconstructed receiver image if the transfer passes CRC and SHA-256. A corrupted packet is rejected; actual retransmission/ACK handling is **not implemented**. The receiver image shown in the UI may be downscaled **for browser display only**, whereas source bytes stay unchanged.

### Packet layout

| Field | Bytes |
|---|---:|
| Magic `HD` | 2 |
| Version | 1 |
| Transfer/session id | 4 |
| Sequence index | 4 |
| Total frame count | 4 |
| Valid chunk length | 1 |
| File data chunk | 0–46 |
| Zero padding to 46 bytes | 0–45 |
| CRC-16 | 2 |
| **Total** | **64** |

`hexadepth/file_transport.py` is deliberately self-contained. `POST /api/lossless` accepts `sample` or `file`, `channels` (1,2,4), `rate_bps`, and `corrupt`. It returns file hashes, packet counts, estimated airtime and receiver display information. Original bytes for browser download are returned only when the received file is ≤5 MiB.

## Concrete benchmark (bundled MILCO real sonar example)

On `0440_2018.jpg` (a bundled real side-scan sonar JPEG), measured locally:

- Source: **328,136 bytes**; zlib content: **326,670 bytes** (only about 0.45% smaller; JPEG is already compressed).
- File envelope: 326,711 bytes; **7,103 packets** of 64 bytes each; **454,592 bytes on wire** under this application framing, excluding modem/FEC overhead.
- At assumed **9,600 bps per independent link**, modeled airtime including assumed fixed 0.48-second setup+propagation: **379.3 s on one link**, **95.2 s on four links**.
- Recovered file SHA-256 matched the original exactly. **None** of these timings is a field measurement; these are idealized arithmetic estimates with fixed link rates and no interference, loss, ACKs, modulation headers, or modem retransmissions.

**Crucial consequence:** splitting into 64-byte frames adds per-frame overhead, often increasing total transmitted bytes even though the file-level codec is lossless. Four separate links may improve throughput *if* they genuinely provide additional acoustic capacity and do not interfere. Four threads on a single modem will not give four times its physical bandwidth. Some modems limit packet size or require their own framing and duty cycle. Sonar transducers do not automatically support arbitrary data communications.

## What to build after the prototype

- **Retransmission (ARQ):** acknowledgement/negative-acknowledgement, timeout, selected retry, frame-level FEC as supported by the modem, and end-to-end file hash validation.
- **Real modem adapter:** serial/Ethernet modem-specific API, link discovery, half-duplex scheduling, maximum allowed application payload, bit rate measurements, latency and error telemetry.
- **Transport scheduling:** bandwidth-aware send queues, lane failure fallback, congestion and interference controls; benchmark whether multi-link throughput gains survive real channel contention.
- **Progressive transfer:** first send a tiny metadata/preview packet, then region-of-interest crops or prioritized image tiles **losslessly**, with full image requested when necessary.
- **AI object descriptions:** transmit small semantic detections only as an optional, explicitly non-image-lossless fallback. An encoded polygon may be bit-perfect *as metadata* but cannot guarantee restoration of unseen pixels or a correct object label.

## Test it

```bat
python -m pytest -q
```

See `tests/test_lossless_transfer.py` for exact file-byte roundtrip, packet sizing, out-of-order delivery, missing/duplicate/corrupt packets, SHA-256 mismatch and real-file API tests. The existing packet and AI modes remain available. The default model-dependent detection capabilities are unchanged; training weights are not bundled in this release.
