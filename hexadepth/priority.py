"""Progressive, priority-first sonar imagery, transported over fixed 64-byte frames.

Three distinct application-layer payloads are sent in sequence:
  1. compact region-of-interest metadata (not a semantic recognition guarantee),
  2. a low-resolution grayscale JPEG covering the entire scene,
  3. small JPEG crops from higher-detail, high-interest regions.

Each payload is actually framed, CRC-checked, reassembled and decoded. The original
file is deliberately NOT included: /api/lossless remains the optional bit-exact
follow-up transfer. Airtime is an estimate and does not simulate acoustic hardware.
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import file_transport as ft
from .comparison import compare, with_metadata

# Stage payloads have compact binary headers so the wire-size estimates include
# every fact needed by a receiver to reconstruct the preview.
META_MAGIC = b"HDP1"
META_HEAD = struct.Struct(">4sHHBB")  # marker, source width, height, count, trained flag
BOX = struct.Struct(">HHHHBB")        # x0, y0, x1, y1, score, class id (255=unknown)
OVERVIEW_HEAD = struct.Struct(">4sHH")
PATCH_HEAD = struct.Struct(">4sBHHHH") # marker, ROI index, x0 y0 width height
MAX_ROIS = 2


class PriorityError(ValueError):
    pass


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    score: int
    class_id: int = 255

    def as_dict(self):
        return {"box": [self.x0, self.y0, self.x1, self.y1],
                "score": self.score, "class_id": None if self.class_id == 255 else self.class_id}


def _shape_score(grey: np.ndarray) -> np.ndarray:
    """Untrained gradient+local-contrast saliency. Not a learned classifier."""
    sm = cv2.GaussianBlur(grey, (5, 5), 0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    mean = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 13)
    contrast = np.abs(sm.astype(np.float32) - mean)
    return edge * .7 + contrast * .3


def saliency_regions(grey: np.ndarray, count: int = MAX_ROIS) -> list[Region]:
    """Pick separated high-texture areas on previously unseen sonar images.

    Every returned label stays 'unverified region' because saliency is not AI
    recognition. No ground truth or training data are accessed.
    """
    h, w = grey.shape
    bw = max(16, min(w, round(w * .28)))
    bh = max(16, min(h, round(h * .28)))
    bw, bh = min(w, bw), min(h, bh)
    sal = _shape_score(grey)
    # Score candidate rectangles on a grid, omitting near-uniform sections.
    step_x, step_y = max(8, bw // 3), max(8, bh // 3)
    boxes = []
    for y in sorted(set(list(range(0, max(1, h - bh + 1), step_y)) + [h - bh])):
        for x in sorted(set(list(range(0, max(1, w - bw + 1), step_x)) + [w - bw])):
            patch = sal[y:y + bh, x:x + bw]
            boxes.append((float(patch.mean()), x, y))
    boxes.sort(reverse=True)
    selected = []
    peak = max((s for s, _, _ in boxes), default=0.)
    if peak < 1.0:
        return []  # uniform scene: do not invent regions of interest
    for score, x, y in boxes:
        if any(max(0, min(x+bw, r.x1)-max(x, r.x0)) *
               max(0, min(y+bh, r.y1)-max(y, r.y0)) > bw*bh*.12 for r in selected):
            continue
        selected.append(Region(x, y, x+bw, y+bh,
                               int(np.clip(round(255 * score / max(peak, 1e-9)), 0, 255))))
        if len(selected) == count:
            break
    return selected


def metadata_pack(w: int, h: int, regions: list[Region], trained: bool) -> bytes:
    if not (1 <= w <= 65535 and 1 <= h <= 65535 and len(regions) <= MAX_ROIS):
        raise PriorityError("Invalid image dimensions or ROI count")
    return META_HEAD.pack(META_MAGIC, w, h, len(regions), 1 if trained else 0) + b"".join(
        BOX.pack(r.x0, r.y0, r.x1, r.y1, r.score, r.class_id) for r in regions)


def metadata_unpack(buf: bytes) -> tuple[int, int, bool, list[Region]]:
    if len(buf) < META_HEAD.size:
        raise PriorityError("Incomplete metadata")
    magic, w, h, n, trained = META_HEAD.unpack_from(buf)
    if magic != META_MAGIC or not w or not h or n > MAX_ROIS or trained not in (0, 1) or len(buf) != META_HEAD.size+n*BOX.size:
        raise PriorityError("Invalid metadata")
    regions = []
    for i in range(n):
        x0, y0, x1, y1, score, cls = BOX.unpack_from(buf, META_HEAD.size+i*BOX.size)
        if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
            raise PriorityError("Out-of-bounds region")
        regions.append(Region(x0,y0,x1,y1,score,cls))
    return w, h, bool(trained), regions


def _imencode_jpeg(grey: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", grey, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise PriorityError("Could not encode image")
    return buf.tobytes()


def _imdecode_jpeg(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise PriorityError("Could not decode received image")
    return decoded


def _transport(payload: bytes, rate_bps: int, corrupt: bool = False) -> tuple[bytes | None, dict]:
    frames = ft.frame_file(payload)
    received = list(frames)
    if corrupt:
        i = len(received)//2
        broken = bytearray(received[i]); broken[ft.HEADER_BYTES] ^= 1
        received[i] = bytes(broken)
    try:
        rebuilt = ft.reassemble_file(received)
        if hashlib.sha256(rebuilt).digest() != hashlib.sha256(payload).digest():
            raise ft.TransferError("Payload integrity mismatch")
        ok, failure = True, None
    except ft.TransferError as exc:
        rebuilt, ok, failure = None, False, str(exc)
    return rebuilt, {"packet_count":len(frames), "wire_bytes":len(frames)*ft.FRAME_BYTES,
                     "payload_bytes":len(payload), "integrity_ok":ok,
                     "error":failure, "airtime_seconds":len(frames)*ft.FRAME_BYTES*8/rate_bps,
                     "first_packet_hex":frames[0].hex().upper()}


def _box_iou(a: Region, b: Region) -> float:
    area = max(0, min(a.x1,b.x1)-max(a.x0,b.x0))*max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
    union = (a.x1-a.x0)*(a.y1-a.y0)+(b.x1-b.x0)*(b.y1-b.y0)-area
    return area/union if union else 0


def select_regions(grey:np.ndarray, detected: list[dict] | None = None) -> tuple[list[Region], bool]:
    """Prefer actual detector boxes when installed; fall back to untrained saliency."""
    h, w = grey.shape
    out=[]
    for d in sorted(detected or [], key=lambda d:-d.get("confidence",0)):
        if "box_px" not in d:
            continue
        x0,y0,x1,y1 = [int(round(v)) for v in d["box_px"]]
        x0,y0 = max(0,x0),max(0,y0)
        x1,y1 = min(w,x1),min(h,y1)
        if x1<=x0 or y1<=y0:
            continue
        pad = max(6,int(.03*min(w,h)))
        r = Region(max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad),
                   int(np.clip(d.get("confidence",0),0,100)*255/100),
                   int(d.get("class_id",255)) if 0<=int(d.get("class_id",255))<=254 else 255)
        if all(_box_iou(r, old)<.3 for old in out):
            out.append(r)
        if len(out)>=MAX_ROIS:
            break
    trained=bool(out)
    if len(out)<MAX_ROIS:
        for r in saliency_regions(grey):
            if all(_box_iou(r,old)<.12 for old in out):
                out.append(r)
            if len(out)>=MAX_ROIS:
                break
    return out,trained


def png_data_url(img:np.ndarray) -> str:
    ok, buf=cv2.imencode(".png",img)
    if not ok:
        raise PriorityError("Could not create preview PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.tobytes()).decode("ascii")


def simulate_progressive(original: bytes, rate_bps:int=2400, detected:list[dict] | None=None,
                         corrupt_stage:str | None=None) -> dict:
    """Complete software-only, sequential progressive transport.

    Full-source transfer is *not* automatically initiated. /api/lossless can
    restore exact original bytes on demand. The receiver image uses only decoded
    payloads from CRC-validated 64-byte frames.
    """
    if not 10<=rate_bps<=62500:
        raise PriorityError("Rate must be between 10 and 62500 bps")
    if corrupt_stage not in (None,"metadata","overview","detail"):
        raise PriorityError("Unknown corruption stage")
    grey=cv2.imdecode(np.frombuffer(original,np.uint8),cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise PriorityError("Not a decodable image")
    h,w=grey.shape
    if max(w,h)>65535:
        raise PriorityError("Image dimensions exceed protocol limit")
    started=time.perf_counter()
    regions,trained=select_regions(grey,detected)
    meta=metadata_pack(w,h,regions,trained)
    # Downscale to 112 pixels maximum for a small full-scene overview.
    factor=min(1.0,112/max(h,w))
    sw,sh=max(1,int(round(w*factor))),max(1,int(round(h*factor)))
    small=cv2.resize(grey,(sw,sh),interpolation=cv2.INTER_AREA)
    overview=OVERVIEW_HEAD.pack(b"HDOV",sw,sh)+_imencode_jpeg(small,35)
    stages=[]
    elapsed=0.48  # a single illustrative setup+propagation delay, not measured
    recon=None
    encoded_regions=[]
    for name,payload in [("metadata",meta),("overview",overview)]:
        received,info=_transport(payload,rate_bps,corrupt_stage==name)
        elapsed += info["airtime_seconds"]
        stage=dict(id=name,title="Priority coordinates" if name=="metadata" else "Whole-scene overview",
                   **info,available_at_seconds=round(elapsed,3))
        if not info["integrity_ok"]:
            stages.append(stage)
            return {"ok":False,"error":stage["error"],"failed_stage":name,"stages":stages,
                    "simulation_only":True}
        if name=="metadata":
            wr,hr,tr,received_regions=metadata_unpack(received)
            assert wr==w and hr==h
        else:
            marker,ow,oh=OVERVIEW_HEAD.unpack_from(received)
            if marker!=b"HDOV":
                raise PriorityError("Overview payload marker invalid")
            image=_imdecode_jpeg(received[OVERVIEW_HEAD.size:])
            if image.shape!=(oh,ow):
                raise PriorityError("Overview geometry invalid")
            recon=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
            stage["image"]=png_data_url(recon)
        stages.append(stage)
    preview=recon.copy()
    overview_mse=float(np.mean((grey.astype(np.float32)-recon.astype(np.float32))**2))
    overview_psnr=round(10*math.log10(255**2/max(overview_mse,1e-10)),2)
    details=[]
    for i,region in enumerate(received_regions):
        # The exact same region crop is encoded on-board and decoded at the receiver.
        # Max patch dimension is 176 px to keep latency manageable.
        crop=grey[region.y0:region.y1,region.x0:region.x1]
        ch,cw=crop.shape
        factor=min(1.0,176/max(ch,cw))
        tw,th=max(1,int(round(cw*factor))),max(1,int(round(ch*factor)))
        down=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_AREA)
        payload=PATCH_HEAD.pack(b"HDPD",i,region.x0,region.y0,cw,ch)+_imencode_jpeg(down,42)
        details.append(payload)
    # Detail payloads use sequential single-link framing and can be independently CRC checked.
    # All detail packets complete before the receiver displays this stage.
    detail_packets=detail_wire=detail_payload=0
    first_hex=""
    for j,payload in enumerate(details):
        received,info=_transport(payload,rate_bps,corrupt_stage=="detail" and j==0)
        detail_packets+=info["packet_count"];detail_wire+=info["wire_bytes"];detail_payload+=info["payload_bytes"]
        if j==0:first_hex=info["first_packet_hex"]
        if not info["integrity_ok"]:
            elapsed+=detail_wire*8/rate_bps
            stages.append(dict(id="detail",title="Important-area detail",packet_count=detail_packets,
                wire_bytes=detail_wire,payload_bytes=detail_payload,integrity_ok=False,
                error=info["error"],airtime_seconds=detail_wire*8/rate_bps,
                available_at_seconds=round(elapsed,3)))
            return {"ok":False,"error":info["error"],"failed_stage":"detail","stages":stages,
                    "simulation_only":True}
        region = received_regions[j]
        marker,idx,x,y,pw,ph=PATCH_HEAD.unpack_from(received)
        if marker!=b"HDPD" or idx!=j or (x,y,pw,ph)!=(region.x0,region.y0,region.x1-region.x0,region.y1-region.y0):
            raise PriorityError("Invalid patch metadata")
        decoded=_imdecode_jpeg(received[PATCH_HEAD.size:])
        preview[y:y+ph,x:x+pw]=cv2.resize(decoded,(pw,ph),interpolation=cv2.INTER_LINEAR)
    elapsed+=detail_wire*8/rate_bps
    detail_mse=float(np.mean((grey.astype(np.float32)-preview.astype(np.float32))**2))
    detail_psnr=round(10*math.log10(255**2/max(detail_mse,1e-10)),2)
    stages.append(dict(id="detail",title="Important-area detail",packet_count=detail_packets,
        wire_bytes=detail_wire,payload_bytes=detail_payload,integrity_ok=True,error=None,
        airtime_seconds=round(detail_wire*8/rate_bps,3),available_at_seconds=round(elapsed,3),
        first_packet_hex=first_hex,image=png_data_url(preview)))
    # Estimate the exact-file follow-up from real file_transfer framing, without
    # sending it or pretending it was requested. Caller may invoke /api/lossless.
    full_envelope,codec=ft.encode_file(original)
    full_packets=math.ceil(len(full_envelope)/ft.PAYLOAD_BYTES)
    full_wire_bytes=full_packets*ft.FRAME_BYTES
    full_airtime=full_wire_bytes*8/rate_bps+.48
    first_info=sum(s["wire_bytes"] for s in stages[:2])
    total_preview=sum(s["wire_bytes"] for s in stages)
    return {"ok":True,"width":w,"height":h,"original_bytes":len(original),
       "model_used":trained,"method":"trained detector with saliency fallback" if trained else "untrained visual saliency (no model weights)",
       "regions":[r.as_dict() for r in received_regions],
       "stages":stages,
       "metadata_and_overview_wire_bytes":first_info,
       "preview_wire_bytes":total_preview,
       "overview_psnr_db":overview_psnr,"detail_psnr_db":detail_psnr,
       "overview_seconds":stages[1]["available_at_seconds"],
       "detail_seconds":stages[2]["available_at_seconds"],
       "full_file_estimate_seconds":round(full_airtime,3),
       "full_file_packets":full_packets,"full_file_codec":codec,
       "estimated_first_view_gain":round(full_airtime/stages[1]["available_at_seconds"],2),
       "rate_bps":rate_bps,"processing_ms":round((time.perf_counter()-started)*1000,2),
       "comparison":with_metadata(compare(first_info,total_preview,full_wire_bytes,rate_bps)),
       "note":"Times are theoretical one-way airtime, using one ideal link and assumed 0.48s startup; no live underwater hardware or model training.",
       "simulation_only":True}
