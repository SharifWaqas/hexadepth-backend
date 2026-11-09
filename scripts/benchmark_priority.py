"""Reproducible measurements for priority-first transmission on bundled real sonar.

Run: python scripts/benchmark_priority.py
No model weights or acoustic modem are required.
"""
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hexadepth.priority import simulate_progressive

OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
rows=[]
for path in sorted((ROOT/'web'/'samples'/'real').glob('*.jpg')):
    r=simulate_progressive(path.read_bytes(),rate_bps=2400)
    if not r['ok']:
        raise RuntimeError(path.name)
    rows.append({
      'sample':path.name,'original_bytes':r['original_bytes'],
      'region_metadata_bytes_on_wire':r['stages'][0]['wire_bytes'],
      'overview_bytes_on_wire':r['metadata_and_overview_wire_bytes'],
      'detail_bytes_on_wire':r['preview_wire_bytes'],
      'region_time_s':r['stages'][0]['available_at_seconds'],
      'overview_time_s':r['overview_seconds'],
      'detail_time_s':r['detail_seconds'],
      'full_file_estimate_s':r['full_file_estimate_seconds'],
      'overview_vs_full_time_ratio':r['estimated_first_view_gain'],
      'overview_psnr_db':r['overview_psnr_db'],
      'detail_psnr_db':r['detail_psnr_db'],
      'trained_model_used':r['model_used'],
    })
with (OUT/'priority_benchmark.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'priority_benchmark.json').write_text(json.dumps(rows,indent=2))
report=["# Priority-first transfer benchmark", "", "This benchmark uses four bundled real MILCO sonar JPEGs, actual OpenCV JPEG encoding, real 64-byte CRC-framed software packetization and reassembly, and a **modeled** ideal single acoustic data link of 2,400 bits/second. No military or commercial underwater modem hardware was operated. Actual trained model weights are absent; regions were chosen by untrained image saliency.", "", "A first overview is lossy and does not preserve fine details. Full-file estimates include the original-byte preserving envelope and 64-byte frames; the complete file is **not** transmitted until requested.", "", "| Sonar file | Original | First overview | Priority detail | Exact original (standalone) | First-view ratio | Overview / detail PSNR |", "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
 report.append(f"| {r['sample']} | {r['original_bytes']/1000:.1f} kB | {r['overview_time_s']:.1f} s | {r['detail_time_s']:.1f} s | {r['full_file_estimate_s']/60:.1f} min | {r['overview_vs_full_time_ratio']:.1f}× | {r['overview_psnr_db']:.1f} / {r['detail_psnr_db']:.1f} dB |")
report.extend(["", "## Interpretation", "", "- First-view improvement compares time to **any coarse scene view** with the time to an entire exact file. It is **not** a speedup for lossless image transfer or a claim of equivalent image quality.", "- Region metadata is 30 bytes and fits within one 64-byte frame; it carries positions and heuristic saliency scores, not proof of object identity.", "- The preview JPEG and ROI patches are encoded with losses; their 64-byte packet transport is CRC checked.", "- Channel airtime assumes 2,400 usable payload bits/s on one independent link plus a single hypothetical 0.48-s setup/propagation delay. It excludes real modem framing, FEC, packet loss, ACK/retransmit, channel contention and processing in vehicle hardware.", "- A trained detector can be installed in the `models/` directory but its cross-dataset generalization must be independently evaluated. This benchmark has **not** trained such a model.", "", "Source: bundled MILCO sonar dataset (Santos et al., 2024), CC BY 4.0. See `web/samples/real/README.md`."])
(OUT/'priority_benchmark.md').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
