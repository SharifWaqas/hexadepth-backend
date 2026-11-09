"""
HexaDepth - Topside Command Dashboard (Module 3).

Run:  streamlit run app.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from hexadepth import packet as pk  # noqa: E402
from hexadepth import synth, topside  # noqa: E402
from hexadepth.vision import VisionParams, process  # noqa: E402

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
PITCH_RAW_BYTES = 2_000_000  # the "2 MB" reference used on the pitch slide

st.set_page_config(page_title="HexaDepth", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def ensure_samples() -> list[str]:
    if not SAMPLES.exists() or not list(SAMPLES.glob("*.png")):
        synth.write_samples(SAMPLES)
    return [p.name for p in sorted(SAMPLES.glob("*.png"))]


@st.cache_data(show_spinner=False)
def run_vision(image_bytes: bytes, params: dict):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return process(img, VisionParams(**params))


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def comparison_chart(raw_s: float, pkt_s: float):
    """Two-bar, log-scale transmission time. Dark surface to match the dashboard."""
    surface, ink, ink2, grid = "#0e1117", "#f0efec", "#c3c2b7", "#2a2d33"
    fig, ax = plt.subplots(figsize=(5, 1.9), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    labels = ["Raw image", "HexaDepth"]
    vals = [raw_s, pkt_s]
    colors = ["#6b6a66", "#3987e5"]
    bars = ax.barh(labels, vals, color=colors, height=0.5)
    ax.set_xscale("log")
    ax.set_xlim(1, max(raw_s, 10) * 30)
    for bar, v in zip(bars, vals):
        ax.text(v * 1.3, bar.get_y() + bar.get_height() / 2, fmt_duration(v),
                va="center", ha="left", color=ink, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.set_xlabel("Transmission time (seconds, log scale)", color=ink2, fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
sample_names = ensure_samples()
if "timestamp" not in st.session_state:
    st.session_state.timestamp = int(time.time())

with st.sidebar:
    st.header("Mission settings")
    uuv_id = st.number_input("UUV ID", 0, 65535, 620)
    drone_x = st.number_input("Vehicle easting (m from datum)", 0, 65535, 12_450)
    drone_y = st.number_input("Vehicle northing (m from datum)", 0, 65535, 30_870)
    depth = st.number_input("Vehicle depth (m)", 0.0, 6553.5, 18.5, step=0.5)
    mpp = st.number_input("Sonar resolution (m per pixel)", 0.005, 1.0, 0.05, step=0.005, format="%.3f")

    with st.expander("Vision tuning"):
        eps = st.slider("RDP epsilon (fraction of perimeter)", 0.005, 0.06, 0.02, 0.005)
        min_area = st.slider("Min target area (fraction of image)", 0.0001, 0.01, 0.0008, 0.0001, format="%.4f")
        min_contrast = st.slider("Min target contrast (× seafloor)", 1.0, 3.0, 1.5, 0.1)
        mask_nadir = st.checkbox("Mask nadir water column", True)

    st.caption("Vehicle is assumed at the image centre, heading north. "
               "Position units: metres from a local survey datum.")

params = dict(meters_per_pixel=mpp, epsilon_frac=eps, min_area_frac=min_area,
              min_contrast=min_contrast, mask_nadir=mask_nadir)

st.title("🌊 HexaDepth")
st.caption("Edge sonar compression: a 2 MB side-scan image becomes a 64-byte acoustic packet.")

col1, col2, col3 = st.columns(3, gap="large")

# ---------------------------------------------------------------- column 1: drone
with col1:
    st.subheader("1 · Submerged view (drone)")
    source = st.selectbox("Sonar frame", sample_names + ["Upload my own…"])
    if source == "Upload my own…":
        up = st.file_uploader("Side-scan image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        image_bytes, label = (up.getvalue(), up.name) if up else (None, None)
    else:
        image_bytes, label = (SAMPLES / source).read_bytes(), source

    tx_hex = None
    raw_bytes = PITCH_RAW_BYTES
    if image_bytes is None:
        st.info("Upload an image to begin.")
    else:
        t0 = time.perf_counter()
        res = run_vision(image_bytes, params)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if res is None:
            st.error("Could not read that image file.")
        else:
            raw_bytes = res.raw.shape[0] * res.raw.shape[1]
            st.image(res.raw, caption=f"Raw sonar frame · {res.raw.shape[1]}×{res.raw.shape[0]} px "
                                      f"· {raw_bytes:,} bytes", width="stretch")
            t1, t2 = st.tabs(["Threshold", "Extracted polygon"])
            with t1:
                st.image(res.threshold, caption="Denoised → range-normalised → Otsu",
                         width="stretch")
            with t2:
                st.image(cv2.cvtColor(res.overlay, cv2.COLOR_BGR2RGB),
                         caption="Orange: raw contour · Green: RDP polygon",
                         width="stretch")

            if not res.found:
                st.warning(res.message)
            else:
                st.success(f"{res.raw_contour_points} contour points → "
                           f"**{len(res.grid_vertices)} vertices** ({elapsed_ms:.0f} ms)")
                det = pk.Detection(
                    uuv_id=int(uuv_id), timestamp=st.session_state.timestamp,
                    drone_x_m=int(drone_x), drone_y_m=int(drone_y), drone_depth_m=float(depth),
                    obj_dx_m=res.obj_dx_m, obj_dy_m=res.obj_dy_m, extent_m=res.extent_m,
                    class_id=res.class_id, confidence=res.confidence,
                    vertices=res.grid_vertices,
                )
                try:
                    tx_hex = pk.to_hex(pk.pack(det))
                except pk.PacketError as e:
                    st.error(f"Cannot pack detection: {e}")
                if tx_hex:
                    st.markdown("**Transmitted acoustic packet (64 bytes)**")
                    st.code(" ".join(tx_hex[i:i + 8] for i in range(0, len(tx_hex), 8)),
                            language=None, wrap_lines=True)

# ---------------------------------------------------------------- column 2: link
with col2:
    st.subheader("2 · Acoustic link (simulation)")
    baud = st.slider("Acoustic baud rate (bps)", 100, 2400, 100, 100)
    raw_s = raw_bytes * 8 / baud
    pkt_s = pk.PACKET_SIZE * 8 / baud

    m1, m2 = st.columns(2)
    m1.metric("Raw image", f"{raw_bytes:,} B")
    m2.metric("HexaDepth packet", f"{pk.PACKET_SIZE} B",
              delta=f"-{(1 - pk.PACKET_SIZE / raw_bytes) * 100:.4f}%", delta_color="inverse")
    m3, m4 = st.columns(2)
    m3.metric("Raw image transmit time", fmt_duration(raw_s))
    m4.metric("HexaDepth transmit time", fmt_duration(pkt_s))
    st.metric("Speed-up", f"{raw_bytes / pk.PACKET_SIZE:,.0f}× faster")

    fig = comparison_chart(raw_s, pkt_s)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("**Channel conditions**")
    n_errors = st.select_slider("Bit errors injected by acoustic interference",
                                options=[0, 1, 2, 4, 8], value=0)
    realtime = st.checkbox("Animate at true link speed", value=True)

    if st.button("📡 Transmit packet", type="primary", disabled=tx_hex is None, width="stretch"):
        bar = st.progress(0.0, text="Transmitting…")
        duration = min(pkt_s, 8.0) if realtime else 1.0
        steps = 50
        for i in range(steps):
            time.sleep(duration / steps)
            bar.progress((i + 1) / steps, text=f"Transmitting… {int((i + 1) / steps * 512)}/512 bits")
        rx = pk.from_hex(tx_hex)
        flipped = random.sample(range(pk.PACKET_SIZE * 8), n_errors)
        for b in flipped:
            rx = pk.flip_bit(rx, b)
        st.session_state.rx_text = pk.to_hex(rx)
        st.session_state.rx_for = tx_hex
        st.session_state.rx_flipped = sorted(flipped)
        bar.progress(1.0, text=f"Delivered in {fmt_duration(pkt_s)} at {baud} bps")

    if tx_hex and st.session_state.get("rx_for") not in (None, tx_hex):
        st.info("A new packet is ready. Press **Transmit** to send it topside.")

# ---------------------------------------------------------------- column 3: ship
with col3:
    st.subheader("3 · Topside view (ship)")
    st.caption("Decodes ONLY the received hex string. Edit it by hand to test the CRC.")
    st.session_state.setdefault("rx_text", "")
    rx_text = st.text_area("Received packet (hex)", key="rx_text", height=110)

    if not rx_text.strip():
        st.info("Waiting for a packet from the vehicle…")
    else:
        flipped = st.session_state.get("rx_flipped") or []
        if flipped:
            st.caption(f"Interference flipped bit(s): {', '.join(map(str, flipped))}")
        try:
            rx_det = pk.unpack(pk.from_hex(rx_text))
        except pk.PacketError as e:
            st.error(f"❌ Packet rejected: {e}")
            st.markdown("Topside would request a retransmit (64 bytes, a few seconds) "
                        "instead of acting on corrupted data.")
        else:
            st.success("✅ CRC-16 verified: packet intact")
            screen = topside.render(rx_det)
            st.image(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB), width="stretch")
            st.markdown(
                f"| Field | Value |\n|---|---|\n"
                f"| Classification | {rx_det.class_name} (ID {rx_det.class_id}) |\n"
                f"| Confidence | {rx_det.confidence}% |\n"
                f"| Target size | {rx_det.extent_m:.2f} m (longest side) |\n"
                f"| Target position | E {rx_det.object_x_m:.1f} m, N {rx_det.object_y_m:.1f} m |\n"
                f"| Vehicle position | E {rx_det.drone_x_m} m, N {rx_det.drone_y_m} m, "
                f"depth {rx_det.drone_depth_m:.1f} m |\n"
                f"| Vertices | {len(rx_det.vertices)} |\n"
            )
            with st.expander("Decoded vertex list (0-255 grid)"):
                st.write(rx_det.vertices)
