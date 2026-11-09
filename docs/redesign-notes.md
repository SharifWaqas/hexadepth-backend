# HexaDepth — November 2026 design and engineering handoff

## Goal and user problem

The old interface made detector classification the starting point. On scenes not represented by the training data, the classifier may miss an object or return an uncertain class. Its confidence score **is not the same thing as validated accuracy**. The mission-critical observation is that an underwater operator needs *some* timely scene information without hauling the vehicle to the surface or sending the full sonar frame.

## Implemented in this update

1. `hexadepth/preview.py`: takes any 8-bit grayscale image, averages it down to 8×8, then quantizes each cell to one bit using the image's intensity percentiles. The exact payload is **64 bits = 8 bytes**. The receiver reverses packing and shows those 64 bits as an enlarged 8×8 black/white map. No learned object categories and no weights are needed. This is a **very coarse visual representation**; details cannot be recovered. High dynamic-range sonar bands may dominate it, and **it cannot be used to assert an object is present or absent**.
2. `hexadepth/link.py`: turns the 8-byte preview or old 64-byte object packet into 1–4 fragments. Each fragment has a **3-byte header** (version, index, count) and **2-byte CRC-16**. It reassembles fragments out of order and rejects corrupt, missing or duplicate fragments. The model assumes physically independent simultaneous acoustic links, with a fixed per-link bit rate, setup = 0.15 s, propagation = 0.33 s. These constants are illustrative. It **does not** implement real acoustic modulation, hardware synchronization or actual underwater communications.
3. `server.py`: adds `POST /api/preview` and `POST /api/link/simulate`. All pre-existing detector/transmission/decode endpoints remain available.
4. `web/index.html`, `web/style.css`, `web/app.js`: rebuilt as a quieter corporate research interface. A single task flows image selection → encode → compare source vs received view → inspect channel lanes. Technical details are collapsed by default.
5. Automated coverage for both modes, real sample image support, 64-bit size guarantees, corruption rejection, fragment ordering and parallel-link timing. Legacy tests remain passing.

## Why 64 bits does not magically solve object recognition

A 1024×1024 grayscale sonar frame contains 8,388,608 uncompressed bits. Encoding it as exactly 64 bits requires discarding 99.99924% of those bits. The receiver can get an *8×8 binary sketch*, but cannot recreate the original features, identify an arbitrary object, or distinguish all materially different scenes. This explicitly **bypasses the need for a classifier** rather than improving classifier generalization. The 64-byte detector-based format is still available to send geometry and class estimates when appropriate.

## Channels and transmission assumptions

- The 8-byte preview uses 13 bytes on one framed channel, or 28 bytes across four framed channels. A 64-byte legacy detection uses 69 bytes on one channel, or 84 across four channels. Payload size and physical on-wire size are not interchangeable.
- At 100 bits/s and the chosen 0.48 s fixed delay assumptions, an 8-byte preview takes about 1.52 s with one link and 1.04 s with four independent links. These are **model outputs only**, not verified hardware performance.
- A single acoustic modem carrying fragments in software "parallel" does not increase the acoustic channel's capacity. Real concurrency requires frequency/code/time allocation or physically distinct independent paths, and can introduce interference and contention. The demo treats the lanes as separately available bit-rate budgets.
- **Sonar** measures the scene. An underwater **acoustic modem** sends digital information. Directly transmitting arbitrary data over an off-the-shelf imaging sonar requires vendor-specific capabilities and is not demonstrated here.
- CRC detects accidental corruption; it does not recover missing data. A practical implementation needs sequence IDs across messages, synchronization, retransmission policy/FEC, queue scheduling, link metrics and hardware integration. The simulator only models one message.

## Visual design decisions

Applied the guidance in [Anthropic's frontend-design skill](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/frontend-design/skills/frontend-design/SKILL.md) as a design reference rather than installing a Claude runtime plugin. Subject: hydrographic and autonomous vessel operations. Audience: researchers, operators, hackathon judges.

- **Palette:** deep sonar navy `#102C3D`, research teal `#087782`, ink `#122F42`, fog gray `#F3F7FA`, white `#FFFFFF`, line `#DBE5EB`.
- **Type:** DM Sans for readable technical UX; IBM Plex Mono sparingly for telemetry/numeric labels; system fallbacks when offline.
- **Layout:** single two-column source/receiver workspace, small controls strip, time/packet summary, collapsible engineering details; stacks on mobile.
- **Distinctive detail:** simplified sonar graticule and target outline in the hero; visual hierarchy prioritizes the operator's before/after view rather than generic KPI cards or decoration.

## Next steps that are **not** implemented

1. **Better scene quality:** test a 64-byte model-free thumbnail/edge preview, a multi-resolution progressive codec and a configurable rate–distortion tradeoff. For an 8-byte preview, compare 8×8 threshold maps against 4×4 4-bit grayscale and 8×8 feature/anomaly maps on a diverse test set.
2. **Measured downstream utility:** assess whether humans can interpret transmitted previews on held-out sonar datasets using blinded tests. Collect appropriate image pairs and quantify what visual information survives at 8, 64, 256 and 1024 bytes. Do not use classification confidence as a proxy for interpretability.
3. **Optional model upgrade:** obtain diverse annotated sonar datasets with access/licensing checks; evaluate trained models strictly on never-seen geographic regions, sensors, and years, add calibration/abstention, and benchmark real accuracy by class.
4. **Hardware proof:** choose a supported commercial acoustic modem, confirm available message sizes, data rates, half-duplex constraints, and whether parallel independent channels are truly possible. Measure airtime, retries, power draw, latency, and packet loss underwater.

## Run and test

From the directory containing `server.py` and `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
python -m pytest -q
```

The web UI is at `http://localhost:8000`.
