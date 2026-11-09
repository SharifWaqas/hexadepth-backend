# Priority-first transfer benchmark

This benchmark uses four bundled real MILCO sonar JPEGs, actual OpenCV JPEG encoding, real 64-byte CRC-framed software packetization and reassembly, and a **modeled** ideal single acoustic data link of 2,400 bits/second. No military or commercial underwater modem hardware was operated. Actual trained model weights are absent; regions were chosen by untrained image saliency.

A first overview is lossy and does not preserve fine details. Full-file estimates include the original-byte preserving envelope and 64-byte frames; the complete file is **not** transmitted until requested.

| Sonar file | Original | First overview | Priority detail | Exact original (standalone) | First-view ratio | Overview / detail PSNR |
|---|---:|---:|---:|---:|---:|---:|
| 0001_2021.jpg | 35.1 kB | 7.1 s | 25.7 s | 2.7 min | 22.9× | 24.7 / 26.6 dB |
| 0009_2017.jpg | 342.1 kB | 7.1 s | 52.5 s | 26.4 min | 223.1× | 19.2 / 19.5 dB |
| 0307_2010.jpg | 312.5 kB | 6.9 s | 51.5 s | 24.1 min | 210.1× | 19.9 / 20.1 dB |
| 0440_2018.jpg | 328.1 kB | 7.3 s | 43.8 s | 25.3 min | 207.4× | 19.6 / 20.1 dB |

## Interpretation

- First-view improvement compares time to **any coarse scene view** with the time to an entire exact file. It is **not** a speedup for lossless image transfer or a claim of equivalent image quality.
- Region metadata is 30 bytes and fits within one 64-byte frame; it carries positions and heuristic saliency scores, not proof of object identity.
- The preview JPEG and ROI patches are encoded with losses; their 64-byte packet transport is CRC checked.
- Channel airtime assumes 2,400 usable payload bits/s on one independent link plus a single hypothetical 0.48-s setup/propagation delay. It excludes real modem framing, FEC, packet loss, ACK/retransmit, channel contention and processing in vehicle hardware.
- A trained detector can be installed in the `models/` directory but its cross-dataset generalization must be independently evaluated. This benchmark has **not** trained such a model.

Source: bundled MILCO sonar dataset (Santos et al., 2024), CC BY 4.0. See `web/samples/real/README.md`.
