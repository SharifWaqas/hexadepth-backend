# HexaDepth — Professional Underwater Sonar Dashboard

## Design Objective

Design a premium, modern, professional web dashboard for **HexaDepth**, an underwater sonar data compression and reconstruction system.

HexaDepth transforms large underwater sonar images into tiny geometric packets that can be transmitted through slow underwater acoustic communication links.

The dashboard will be demonstrated to hackathon judges, investors, marine researchers, and nontechnical audiences.

**The most important requirement:** Anyone looking at the dashboard should understand what HexaDepth does within 10 seconds, even without knowing anything about sonar, computer vision, or underwater robotics.

The interface should feel like a real maritime technology command center, not a student project or a complicated developer tool.

## 1. Visual Identity

**Style:** Modern ocean technology, scientific visualization, premium SaaS dashboard, with subtle naval command-center influences.

**Color palette — "Side-scan Bronze":**
- Page background: Blue-black `#0B0F14`
- Cards / secondary surfaces: `#141A21`
- Borders / dividers: `#232C36`
- Primary accent (buttons, active states, links, sliders, progress bars, the travelling packet, highlights): Copper `#D98A4E`
  - Hover: Light copper `#E39A5F`; text on copper buttons: `#0B0F14`
- Primary text: Warm white `#F2EDE6`
- Secondary text: `#9AA3AD`
- Success / Verified / detected-object outline: Green `#4CC38A`
- Error / Corrupted / Rejected: Red `#E5484D` (error *text* uses `#EF5F63` to meet WCAG AA inside tinted badges)
- Warning / in-progress states: Gold `#E8B04B` (kept clearly distinct from the copper accent)
- Traditional-transmission bar: Muted grey-brown `#5A4A3C`
- Sonar imagery: grey levels mapped to a bronze ramp, dark brown `#120A05` → copper `#D98A4E` → warm gold `#F6D7A7`, as in real side-scan software. The detected-object outline stays green so it stands out.

All colours are defined once as CSS custom properties at the top of `web/style.css`.

**Typography:** Inter or Geist.

Use sophisticated typography, subtle gradients, rounded cards, thin borders, excellent spacing, simple line icons, and restrained animations.

Avoid excessive neon effects, crowded technical panels, unnecessary charts, and complex engineering terminology.

## 2. Overall Dashboard Layout

Design a responsive desktop-first dashboard at 1440px width.

### Top Navigation Bar

Left:
- HexaDepth logo (minimal sonar wave combined with geometric polygon)
- Product name: **HexaDepth**
- Subtitle: *Underwater Intelligence, Simplified*

Center navigation:
- Live Demo
- How It Works
- Technical Details

Right:
- Status indicator: "Simulation Ready"
- Help / About icon

### Hero Section

Headline:

**See More. Send Less.**

Subheadline:

"Turn massive underwater sonar images into tiny data packets, so ships can understand what's underwater without waiting hours for an image."

Include a small explanatory label:

**Underwater Drone → Acoustic Signal → Surface Ship**

Two actions:
- **Try the Demo** (primary)
- **How It Works** (secondary)

Do not let the hero consume too much vertical space. The interactive demo should remain the primary focus.

## 3. Performance Overview

Immediately beneath the hero, add four clean metric cards.

| Metric | Value |
|---|---|
| Original Image | 2 MB |
| HexaDepth Packet | 64 Bytes |
| Data Reduction | 99.9968% |
| Simulated Delivery | 5.12 Seconds |

Use subtle count-up animations during a demo.

Include a short explanation below:

"At an ideal 100 bits per second, sending an entire 2 MB image takes over 44 hours. HexaDepth sends a simplified object outline in about 5 seconds."

Make it clear that these are illustrative transmission calculations and that HexaDepth sends a geometric summary, not the full original image.

## 4. Main Interactive Demo — The Centerpiece

Create a spacious three-stage dashboard.

At the top, display a horizontal process indicator:

**1. Detect → 2. Transmit → 3. Reconstruct**

Connect the stages with subtle animated lines showing data moving from left to right.

### Panel 1: Underwater Drone

**Title:** What the Drone Sees

**Subtitle:** "Discover an object hidden beneath the water."

Include:
- Large sonar image viewing area
- Upload Image button
- Sample Sonar Image dropdown
- Original / Processed image toggle
- Clear green polygon outlining the detected object
- Subtle animated sonar sweep
- Detection confidence indicator
- Small "Object Detected" success status

Show the three computer-vision processing stages in an optional expandable section:
- Noise Removal
- Object Detection
- Shape Simplification

Add a prominent action:

**Analyze Sonar Image**

After activation, animate the raw image transitioning into a simplified polygon.

Do not display raw OpenCV logs or coordinate arrays in the primary view.

### Panel 2: Acoustic Transmission

**Title:** Sending Through Water

**Subtitle:** "Only the object's essential shape is transmitted."

This panel should visually explain the main innovation.

Create a vertical before-and-after comparison:

**Traditional Image Transmission**
- 2 MB
- Approximately 44.4 hours at 100 bps
- Long grey-brown progress visualization

**HexaDepth Transmission**
- 64 Bytes
- Approximately 5.12 seconds at 100 bps
- Short copper progress visualization

Between the comparisons, add:

**31,250× Smaller**

Below, display:
- Acoustic Data Rate slider (100–2400 bps)
- Dynamic estimated transmission times
- Transmission progress indicator
- Packet Sent / Packet Received statuses

Add a visually styled expandable section titled:

**View Transmitted Data**

Display the hexadecimal representation in a monospace panel with a Copy button.

Include a tooltip explaining that the hex is a readable representation of a binary packet, not additional data sent over the underwater link.

### Panel 3: Surface Ship

**Title:** What the Ship Receives

**Subtitle:** "Rebuild a useful outline from just 64 bytes."

Create a large reconstructed-sonar display:
- Dark navy background
- Faint copper sonar grid
- Reconstructed green polygon
- Animated outline-drawing effect
- Object marker and simple coordinate axes

Below the image, show:
- Object Type
- Detection Confidence
- Drone Position (X, Y, Z)
- Packet Integrity: Verified

Include a small explanation:

"The ship receives a compact description of the object's shape, not the original photograph."

Add a **Compare Original & Reconstruction** control, allowing users to see both visuals side by side.

## 5. Demo Controls

Place a small, visually distinct control strip directly above the three panels.

Include:
- Select Sample
- Upload Sonar Image
- Run Simulation
- Reset Demo

Use a simple guided state progression:

**Ready → Analyzing → Transmitting → Reconstructed**

Each stage should visibly update its status as the simulation advances.

Provide a clear demonstration experience with minimal user interaction. One click should be enough to show the entire process using a preloaded sample image.

Avoid forcing the judges to configure technical parameters.

## 6. How HexaDepth Works

Below the live demo, create a clean educational section titled:

**From Underwater Image to Surface Intelligence**

Use four horizontal illustrated cards:

**01 — Capture**

An underwater drone collects a sonar image.

**02 — Simplify**

HexaDepth identifies the object's outline and reduces it to essential points.

**03 — Transmit**

The compact geometric packet travels through a simulated acoustic communication link.

**04 — Reconstruct**

The ship recreates the object's approximate shape from the received data.

Each card should use a simple icon, one sentence, and a subtle connecting line.

## 7. Optional Technical Details

Create a collapsed section called **Explore the Engineering**.

This content is for technical judges and marine engineers. Keep it hidden by default.

Use four technical cards:
- **Computer Vision:** OpenCV filtering, contour extraction, polygon simplification
- **Binary Encoding:** Compact packet structure, normalized coordinates, CRC-16
- **Underwater Communication:** Acoustic modem simulation and data-rate calculations
- **Vehicle Integration Concept:** Isolated edge payload processor, sonar ingestion, modem interface

Include a small binary packet diagram and an illustrative system architecture.

Label external vehicle and modem integration as **proposed**, not as hardware-verified functionality.

## 8. Interaction and Motion

Animations should communicate the technology rather than distract from it.

Use:
- Slow sonar sweeping animation in the image viewer
- Smooth contour extraction and polygon simplification
- Small moving packet animation between drone and ship
- Subtle progress animation during transmission
- Polygon reconstruction drawing itself on the ship display
- Smooth metric transitions
- Gentle hover states and tooltips

Keep animations professional and restrained. Avoid dramatic sci-fi effects.

## 9. Accessibility and Navigation

Make the interface understandable to people with no engineering background.

Use plain-language headings before technical terminology.

Examples:
- "What the Drone Sees" instead of "Raw Sonar Ingestion"
- "Sending Through Water" instead of "Binary Packet Transport"
- "What the Ship Receives" instead of "Polygon Decoding"

Provide short tooltips for:
- Sonar
- Acoustic Transmission
- Data Reduction
- Detection Confidence
- Reconstructed Outline

Use large readable text, strong contrast, clear action buttons, accessible status labels, and a consistent visual hierarchy.

## 10. Responsive Design

Desktop:
- Three-panel demo layout
- Performance metrics in a horizontal row
- Wide sonar visualization displays

Tablet:
- Two-column or stacked panels, preserving the detection-to-reconstruction sequence

Mobile:
- One panel per step
- Large next-step buttons
- Simplified metrics and collapsible technical sections

## Final Design Direction

The finished design should resemble a polished combination of a high-end maritime monitoring system and a modern AI analytics platform.

It should be visually impressive but extremely intuitive.

The primary visual story is:

**A large sonar image enters. A tiny packet travels through water. A recognizable object outline appears on the ship's screen.**

Prioritize the interactive demonstration, dramatic data-size comparison, clean information hierarchy, and clear storytelling.

**Design the complete frontend layout and interaction states only. Do not build the backend, implement computer vision algorithms, or invent real sensor connectivity.**

The final result should look investor-ready, hackathon-ready, and suitable for a real ocean technology product.



One important design recommendation: Make the three-panel live demonstration the largest and most visually prominent part of the interface. The strongest moment for the judges will be seeing an underwater image turn into a tiny packet and then reappear as a recognizable outline on the other side.
