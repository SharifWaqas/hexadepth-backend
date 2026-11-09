"""Reproducible HexaDepth software benchmark against PUBLIC underwater-modem rates.

IMPORTANT: This script makes no connections to acoustic or military hardware.
It measures file preparation/reassembly locally and models one-way airtime as
8 * application-framed bytes / specified rate. Nominal vendor rates are not
interchangeable with actual payload goodput; real results need modem testing.

Usage from project root: python scripts/benchmark_public_modems.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hexadepth import file_transport as ft

# Explicitly documented public profiles, not hardware owned or operated here.
# See reference URLs in output documentation.
PROFILES = [
    {
        'system': 'Teledyne Benthos ATM-900 / Gavia (MFSK upper setting)',
        'rate_bps': 1200,
        'rate_type': 'published modulation setting (not end-to-end goodput)',
        'source': 'https://www.teledynemarine.com/brands/gavia/acoustic-modem-option-control-module?ProductLineID=19',
    },
    {
        'system': 'WHOI Micromodem single receiver (documented throughput)',
        'rate_bps': 3343,
        'rate_type': 'best-case continuous throughput example, no ACK',
        'source': 'https://acomms.whoi.edu/micro-modem',
    },
    {
        'system': 'WHOI Micromodem four-channel receiver (documented throughput)',
        'rate_bps': 1622,
        'rate_type': 'receiver diversity, NOT four independent transmissions',
        'source': 'https://acomms.whoi.edu/micro-modem',
    },
    {
        'system': 'EvoLogics 18/34 (advertised peak)',
        'rate_bps': 13900,
        'rate_type': 'peak acoustic link rate, not guaranteed app goodput',
        'source': 'https://www.evologics.com/underwater-acoustic-modem-series/18-34',
    },
    {
        'system': 'Teledyne Benthos ATM-900 / Gavia (optional PSK peak)',
        'rate_bps': 15360,
        'rate_type': 'published optional PSK peak, not end-to-end goodput',
        'source': 'https://www.teledynemarine.com/brands/gavia/acoustic-modem-option-control-module?ProductLineID=19',
    },
]


def benchmark_file(name: str, data: bytes) -> dict:
    # Cold-ish first pass, repeated and summarized to reduce clock jitter.
    encode_ms: list[float] = []
    decode_ms: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        envelope, codec = ft.encode_file(data)
        frames = ft.frame_file(envelope)
        t1 = time.perf_counter()
        restored = ft.decode_file(ft.reassemble_file(frames))
        t2 = time.perf_counter()
        assert restored == data and hashlib.sha256(restored).digest() == hashlib.sha256(data).digest()
        encode_ms.append((t1 - t0) * 1000)
        decode_ms.append((t2 - t1) * 1000)
    # Check reordering is tolerated; an application *could* use parallel links.
    assert ft.decode_file(ft.reassemble_file(frames[::-1])) == data
    # Check at least one corrupted packet rejects, not silently reconstructs.
    broken = list(frames)
    middle = bytearray(broken[len(broken) // 2])
    middle[ft.HEADER_BYTES] ^= 1
    broken[len(broken) // 2] = bytes(middle)
    try:
        ft.decode_file(ft.reassemble_file(broken))
    except ft.TransferError:
        pass
    else:
        raise AssertionError(f'Corruption not detected for {name}')

    compressed_bytes = len(envelope) - ft.FILE_HEADER_BYTES
    raw_framed_packets = math.ceil((len(data) + ft.FILE_HEADER_BYTES) / ft.PAYLOAD_BYTES)
    result = {
        'file': name,
        'source_bytes': len(data),
        'codec': codec,
        'compressed_content_bytes': compressed_bytes,
        'codec_savings_pct': round(100 * (1 - compressed_bytes / len(data)), 3),
        'envelope_bytes': len(envelope),
        'packet_count': len(frames),
        'wire_bytes': sum(map(len, frames)),
        'wire_vs_source_pct': round(100 * (sum(map(len, frames)) / len(data) - 1), 2),
        'uncompressed_64b_wire_bytes': raw_framed_packets * ft.FRAME_BYTES,
        'benefit_vs_same_framing_pct': round(100 * (1 - len(frames) / raw_framed_packets), 3),
        'local_encode_packetize_ms_median': round(statistics.median(encode_ms), 2),
        'local_reassemble_decode_ms_median': round(statistics.median(decode_ms), 2),
        'sha256_ok': True,
        'corruption_rejected': True,
        'reverse_order_ok': True,
    }
    return result


def main() -> None:
    root_samples = ROOT / 'web' / 'samples' / 'real'
    samples = {p.name: p.read_bytes() for p in sorted(root_samples.glob('*.jpg'))}
    if not samples:
        raise RuntimeError('No real sonar JPEG samples were found')
    # Compare with an uncompressed representation of the same visible pixels:
    # This BMP contains no extra sensor information beyond the source JPEG.
    import cv2
    import numpy as np
    input_image = cv2.imdecode(np.frombuffer(samples['0440_2018.jpg'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert input_image is not None
    ok, bmp = cv2.imencode('.bmp', input_image)
    assert ok
    samples['0440_2018_decoded_grayscale.bmp'] = bmp.tobytes()

    file_results = [benchmark_file(name, b) for name, b in samples.items()]
    by_filename = {r['file']: r for r in file_results}
    representative = by_filename['0440_2018.jpg']
    bmp = by_filename['0440_2018_decoded_grayscale.bmp']

    # All four scenarios below are intentionally idealized. Hardware overhead,
    # propagation, acoustic interference, retransmissions and FEC are omitted.
    times = []
    for profile in PROFILES:
        wire = representative['wire_bytes']
        raw = representative['source_bytes']
        speed = profile['rate_bps']
        times.append({
            'system': profile['system'],
            'rate_bps': speed,
            'rate_type': profile['rate_type'],
            'source': profile['source'],
            'original_jpeg_lower_bound_seconds': round(8 * raw / speed, 2),
            'hexadepth_one_link_ideal_seconds': round(8 * wire / speed, 2),
            'hexadepth_one_link_minutes': round(8 * wire / speed / 60, 2),
            'hexadepth_four_truly_independent_links_ideal_seconds': None if 'four-channel receiver' in profile['system'] else round(8 * math.ceil(representative['packet_count']/4) * 64 / speed, 2),
            'same_total_capacity_split_four_ideal_seconds': round(8 * wire / speed, 2),
            'half_advertised_rate_illustration_seconds': round(16 * wire / speed, 2),
        })

    output = ROOT / 'results'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'public_modem_benchmark_files.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=file_results[0].keys())
        writer.writeheader(); writer.writerows(file_results)
    with (output / 'public_modem_benchmark_systems.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=times[0].keys())
        writer.writeheader(); writer.writerows(times)
    (output / 'public_modem_benchmark.json').write_text(json.dumps({'method': 'Software verified locally + link airtime inferred from public rates; NOT an in-water modem benchmark', 'file_results': file_results, 'system_results': times}, indent=2), encoding='utf-8')

    from matplotlib import pyplot as plt
    plotting = [times[i] for i in [0, 1, 3, 4]]
    labels = ['Benthos 1.2 kbps', 'WHOI 3.34 kbps', 'EvoLogics 13.9 kbps', 'Benthos 15.36 kbps']
    ys = [p['hexadepth_one_link_minutes'] for p in plotting]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels[::-1], ys[::-1], color='#147D84')
    ax.set_xlabel('Minutes (ideal one-way airtime at documented link rate)')
    ax.set_title('HexaDepth | 328,136-byte sonar JPEG, 7,103 packets', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(ys) * 1.19)
    ax.spines[['top', 'right']].set_visible(False)
    for b in bars:
        ax.text(b.get_width() + max(ys)*0.016, b.get_y() + b.get_height()/2, f'{b.get_width():.1f} min', va='center', fontsize=9)
    fig.text(0.08, 0.015, 'Simulation only. No underwater hardware tested; actual throughput depends on conditions and modem protocols.', fontsize=8, color='#526271')
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(output / 'public_modem_benchmark.png', dpi=180)
    plt.close(fig)

    def minsec(sec: float) -> str:
        m,s = divmod(round(sec), 60)
        return f'{m}m {s:02d}s'

    md = [
        '# HexaDepth acoustic-modem reference benchmark', '',
        '**Scope:** Local software tests and calculated, idealized transfer times against *public* modem rates. No military/acoustic hardware was accessed, configured, or tested.', '',
        '## What was measured vs modeled', '',
        '- **Measured locally:** lossless compression sizes; exact 64-byte packet counts; CRC/SHA-256 verified reassembly; local CPU timings in the current environment (3-run medians).',
        '- **Modeled:** one-way transmission time `64 bytes × 8 bits × frame count ÷ documented bits/sec` for constant-rate links.',
        '- **Not modeled:** half-duplex polling, modem frame/PHY framing, FEC, inter-packet guard times, time of flight, ACK/retry, sea state, collisions, SNR, spectrum allocation, or export-controlled configurations.',
        '- Nominal modulation rates and measured payload throughputs are **not equivalent**: these scenarios are indicative bounds, not directly comparable hardware performance claims.', '',
        '## Compression and packetization results', '',
        '| Input | Original KiB | Codec | Codec saving | 64-byte frames | Framed KiB | Framing change vs original | Local encode+packetize ms |',
        '|:--|--:|:--|--:|--:|--:|--:|--:|'
    ]
    for r in file_results:
        md.append(f"| {r['file']} | {r['source_bytes']/1024:,.1f} | {r['codec']} | {r['codec_savings_pct']:.2f}% | {r['packet_count']:,} | {r['wire_bytes']/1024:,.1f} | {r['wire_vs_source_pct']:+.1f}% | {r['local_encode_packetize_ms_median']:,.2f} |")
    md += ['', 'The BMP row comes from **decoding the same JPEG** into a grayscale bitmap. It is a codec format comparison, not a distinct sonar sensor capture; it contains no more information than the compressed JPEG.', '',
           '## Representative sonar JPEG, real MILCO sample `0440_2018.jpg`', '',
           f"- Source file: **{representative['source_bytes']:,} bytes**. Lossless zlib content **{representative['compressed_content_bytes']:,} bytes**, only **{representative['codec_savings_pct']:.2f}%** codec savings.",
           f"- 64-byte frames: **{representative['packet_count']:,}**, with 46 bytes payload per frame. Total **{representative['wire_bytes']:,} bytes transmitted at the application-frame level**, **{representative['wire_vs_source_pct']:+.2f}%** against the already-compressed JPEG.",
           f"- CRC corruption rejection, reverse-order packet reassembly, and exact SHA-256 matching: **passed**.", '',
           '### One-way ideal airtime per publicly documented modem profile', '',
           '| Public reference | Rate | Raw JPEG (unframed lower bound) | HexaDepth framed 1 link | HexaDepth framed 4 independent equal-rate links* |',
           '|:--|--:|--:|--:|--:|']
    for x in times:
        md.append(f"| {x['system']} | {x['rate_bps']:,} bps | {minsec(x['original_jpeg_lower_bound_seconds'])} | **{minsec(x['hexadepth_one_link_ideal_seconds'])}** | {minsec(x['hexadepth_four_truly_independent_links_ideal_seconds']) if x['hexadepth_four_truly_independent_links_ideal_seconds'] is not None else 'N/A: multichannel RX, not separate links'} |")
    md += ['', "*4-link numbers are a **hypothetical upper-bound scenario** requiring four genuinely independent full-rate modems/channels, not a supported feature demonstrated on these devices. Splitting a single shared modem's aggregate rate four ways gives **no airtime reduction**. WHOI's documented four-receiver configuration is **slower** than its single-receiver configuration (1,622 vs 3,343 bps), so it MUST NOT be portrayed as four independent links.", '',
           'The raw JPEG column is a lower bound only, without packet framing. A real modem will add its own framing and handling. For apples-to-apples software codec benefit, `uncompressed_64b_wire_bytes` in the CSV uses the **same** HexaDepth framing without zlib.', '',
           '## Findings', '',
           f"- On the representative JPEG, zlib barely helps and 64-byte framing increases byte count by {representative['wire_vs_source_pct']:.1f}% relative to simply counting the JPEG's existing bytes. This software is **not** faster than existing image-file transmission on one link, all else equal.",
           '- The **codec can help on uncompressed, repetitive sources** such as BMP/raw, but when a compressed JPEG or PNG is already available it may not.',
           '- Physical parallelism can help when truly independent additional spectrum/radio-acoustic resources exist, but channel count by itself is not evidence of extra throughput.',
           '- NATO JANUS (STANAG 4748) is an underwater signalling/discovery/interoperability standard, not an appropriate high-throughput image-modem benchmark. It has a 64-bit baseline packet, whereas HexaDepth frames are 64 **bytes**.',
           '- A more promising approach for very fast situational awareness is **send small AI/object metadata first** (when confidence warrants it) and transfer the exact image later or on request, with unknown objects flagged as unknown.', '',
           '## Sources (public documentation)', '',
           '- Teledyne Marine Gavia Benthos modem: https://www.teledynemarine.com/brands/gavia/acoustic-modem-option-control-module?ProductLineID=19',
           '- Teledyne Gavia defense applications: https://www.teledynemarine.com/brands/gavia/gavia-auv?ProductLineID=23',
           '- WHOI Micromodem rates and throughput examples: https://acomms.whoi.edu/micro-modem',
           '- EvoLogics 18/34: https://www.evologics.com/underwater-acoustic-modem-series/18-34',
           '- NATO JANUS STANAG 4748: https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=284318',
           '', '## Reproduce', '', '```console', 'python scripts/benchmark_public_modems.py', 'python -m pytest -q', '```', '',
           'This report and accompanying CSV/JSON/chart are generated by the included script.']
    (output / 'public_modem_benchmark.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'representative': representative, 'bmp': bmp, 'systems': times}, indent=2))
    print('Saved benchmark artifacts to', output)


if __name__ == '__main__':
    main()
