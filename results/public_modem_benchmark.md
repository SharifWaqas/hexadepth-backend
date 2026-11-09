# HexaDepth acoustic-modem reference benchmark

**Scope:** Local software tests and calculated, idealized transfer times against *public* modem rates. No military/acoustic hardware was accessed, configured, or tested.

## What was measured vs modeled

- **Measured locally:** lossless compression sizes; exact 64-byte packet counts; CRC/SHA-256 verified reassembly; local CPU timings in the current environment (3-run medians).
- **Modeled:** one-way transmission time `64 bytes × 8 bits × frame count ÷ documented bits/sec` for constant-rate links.
- **Not modeled:** half-duplex polling, modem frame/PHY framing, FEC, inter-packet guard times, time of flight, ACK/retry, sea state, collisions, SNR, spectrum allocation, or export-controlled configurations.
- Nominal modulation rates and measured payload throughputs are **not equivalent**: these scenarios are indicative bounds, not directly comparable hardware performance claims.

## Compression and packetization results

| Input | Original KiB | Codec | Codec saving | 64-byte frames | Framed KiB | Framing change vs original | Local encode+packetize ms |
|:--|--:|:--|--:|--:|--:|--:|--:|
| 0001_2021.jpg | 34.2 | zlib | 0.70% | 758 | 47.4 | +38.4% | 1.49 |
| 0009_2017.jpg | 334.1 | zlib | 0.32% | 7,415 | 463.4 | +38.7% | 15.98 |
| 0307_2010.jpg | 305.2 | zlib | 0.32% | 6,774 | 423.4 | +38.7% | 10.44 |
| 0440_2018.jpg | 320.4 | zlib | 0.45% | 7,103 | 443.9 | +38.5% | 11.00 |
| 0440_2018_decoded_grayscale.bmp | 1,025.1 | zlib | 20.70% | 18,097 | 1,131.1 | +10.3% | 32.93 |

The BMP row comes from **decoding the same JPEG** into a grayscale bitmap. It is a codec format comparison, not a distinct sonar sensor capture; it contains no more information than the compressed JPEG.

## Representative sonar JPEG, real MILCO sample `0440_2018.jpg`

- Source file: **328,136 bytes**. Lossless zlib content **326,670 bytes**, only **0.45%** codec savings.
- 64-byte frames: **7,103**, with 46 bytes payload per frame. Total **454,592 bytes transmitted at the application-frame level**, **+38.54%** against the already-compressed JPEG.
- CRC corruption rejection, reverse-order packet reassembly, and exact SHA-256 matching: **passed**.

### One-way ideal airtime per publicly documented modem profile

| Public reference | Rate | Raw JPEG (unframed lower bound) | HexaDepth framed 1 link | HexaDepth framed 4 independent equal-rate links* |
|:--|--:|--:|--:|--:|
| Teledyne Benthos ATM-900 / Gavia (MFSK upper setting) | 1,200 bps | 36m 28s | **50m 31s** | 12m 38s |
| WHOI Micromodem single receiver (documented throughput) | 3,343 bps | 13m 05s | **18m 08s** | 4m 32s |
| WHOI Micromodem four-channel receiver (documented throughput) | 1,622 bps | 26m 58s | **37m 22s** | N/A: multichannel RX, not separate links |
| EvoLogics 18/34 (advertised peak) | 13,900 bps | 3m 09s | **4m 22s** | 1m 05s |
| Teledyne Benthos ATM-900 / Gavia (optional PSK peak) | 15,360 bps | 2m 51s | **3m 57s** | 0m 59s |

*4-link numbers are a **hypothetical upper-bound scenario** requiring four genuinely independent full-rate modems/channels, not a supported feature demonstrated on these devices. Splitting a single shared modem's aggregate rate four ways gives **no airtime reduction**. WHOI's documented four-receiver configuration is **slower** than its single-receiver configuration (1,622 vs 3,343 bps), so it MUST NOT be portrayed as four independent links.

The raw JPEG column is a lower bound only, without packet framing. A real modem will add its own framing and handling. For apples-to-apples software codec benefit, `uncompressed_64b_wire_bytes` in the CSV uses the **same** HexaDepth framing without zlib.

## Findings

- On the representative JPEG, zlib barely helps and 64-byte framing increases byte count by 38.5% relative to simply counting the JPEG's existing bytes. This software is **not** faster than existing image-file transmission on one link, all else equal.
- The **codec can help on uncompressed, repetitive sources** such as BMP/raw, but when a compressed JPEG or PNG is already available it may not.
- Physical parallelism can help when truly independent additional spectrum/radio-acoustic resources exist, but channel count by itself is not evidence of extra throughput.
- NATO JANUS (STANAG 4748) is an underwater signalling/discovery/interoperability standard, not an appropriate high-throughput image-modem benchmark. It has a 64-bit baseline packet, whereas HexaDepth frames are 64 **bytes**.
- A more promising approach for very fast situational awareness is **send small AI/object metadata first** (when confidence warrants it) and transfer the exact image later or on request, with unknown objects flagged as unknown.

## Sources (public documentation)

- Teledyne Marine Gavia Benthos modem: https://www.teledynemarine.com/brands/gavia/acoustic-modem-option-control-module?ProductLineID=19
- Teledyne Gavia defense applications: https://www.teledynemarine.com/brands/gavia/gavia-auv?ProductLineID=23
- WHOI Micromodem rates and throughput examples: https://acomms.whoi.edu/micro-modem
- EvoLogics 18/34: https://www.evologics.com/underwater-acoustic-modem-series/18-34
- NATO JANUS STANAG 4748: https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=284318

## Reproduce

```console
python scripts/benchmark_public_modems.py
python -m pytest -q
```

This report and accompanying CSV/JSON/chart are generated by the included script.
