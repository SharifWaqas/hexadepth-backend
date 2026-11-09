"""
Evaluate HexaDepth on real side-scan sonar (MILCO dataset) with an honest tune/test split.

For every image: vision -> 64-byte packet -> decode from the bytes only -> compare the decoded
object's bounding box with the ground-truth YOLO boxes.

    python scripts/evaluate_real.py --split test --detector classical --profile real
    python scripts/evaluate_real.py --split test --detector oracle
    python scripts/evaluate_real.py --split tune --profile default --limit 20

Detectors (pluggable, see DETECTORS below):
    classical  the OpenCV pipeline in hexadepth/vision.py with a named --profile
    oracle     ground-truth boxes stand in for a perfect detector; each box is encoded as its own
               64-byte packet. Measures ONLY what the packet preserves, not detection.
    authors_yolov4  the dataset authors' trained YOLOv4 (Santos et al. 2024, CC BY 4.0), OpenCV DNN, CPU
    yolo            our YOLOv8n fine-tuned on the D2 train split (needs requirements-yolo.txt)
    (see hexadepth/detectors.py; a missing model falls back to the next detector with a note)

    python scripts/evaluate_real.py --compare      # headline: all detectors on d2-test (2018+2021)

--compare writes results/real_eval.json + .md (the comparison the website shows) plus
results/real_eval_d2-test_<detector>.json/.md. Single runs write
results/real_eval_<split>_<detector>[_<profile>][...].json/.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexadepth import packet as pk  # noqa: E402
from hexadepth import realdata as rd  # noqa: E402
from hexadepth import detectors as dets  # noqa: E402
from hexadepth.detectors import DETECTORS  # noqa: E402
from hexadepth.vision import PROFILES, params_for  # noqa: E402

IOU_THRESHOLDS = (0.5, 0.3)
SIZE_MATCH_IOU = 0.3          # size comparison uses objects matched at this IoU (more samples)
JPEG_QUALITIES = (75, 50)
CROP_PAD = 0.10
LINK_BPS = 100

# Fixed mission fields for the packet (as in the web demo); they do not affect the metrics.
MISSION = dict(uuv_id=620, timestamp=1_700_000_000, drone_x_m=12_450, drone_y_m=30_870, drone_depth_m=18.5)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def pct(v):
    return None if v is None else round(100 * v, 1)


def median_or_none(xs):
    return round(float(statistics.median(xs)), 2) if xs else None


def percentile_or_none(xs, q):
    return round(float(np.percentile(xs, q)), 2) if xs else None


def coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


# ---------------------------------------------------------------- detectors
# Shared with the web server: see hexadepth/detectors.py (classical, oracle, authors_yolov4, yolo).
COMPARE_DETECTORS = ("authors_yolov4", "yolo", "classical", "oracle")
COMPARE_SPLIT = "d2-test"


def detector_settings(det) -> dict:
    if det.name == "authors_yolov4":
        return dict(runtime=f"OpenCV {cv2.__version__} DNN (Darknet importer), CPU", cfg=det.cfg.name,
                    weights=det.weights.name, input_size=[det.input_w, det.input_h],
                    confidence_threshold=dets.AUTHORS_CONF, nms_iou=dets.AUTHORS_NMS,
                    threshold_source="Darknet default detection threshold (0.25); not chosen on test images")
    if det.name == "yolo":
        import ultralytics
        return dict(runtime=f"Ultralytics {ultralytics.__version__}, CPU", weights=det.weights.name,
                    imgsz=dets.YOLO_IMGSZ, confidence_threshold=dets.YOLO_CONF, nms_iou=dets.YOLO_NMS,
                    threshold_source="Ultralytics default predict settings; not chosen on test images")
    return {}


def training_overlap(detector: str, split: str) -> str:
    if detector == "authors_yolov4":
        return dets.AUTHORS_TRAINING_NOTE
    if detector == "yolo":
        return ("none" if split == "d2-test" else
                "trained on d2-train (2010+2015+2017 minus the val slice); only d2-test is held out")
    if detector == "classical":
        return "none" if split in ("test", "d2-test") else "real profile tuned on 2015"
    return "n/a (ground truth)"


def evaluate(root: Path, split: str, profile: str, limit: int | None, overrides: dict, quiet: bool = False,
             detector: str = "classical", fallback: bool = True) -> dict:
    """Run one detector over a split. A detector whose model is missing falls back to the next
    available one (authors_yolov4 -> yolo -> classical) with a visible note, unless fallback=False
    (then ModelUnavailable is raised)."""
    names = rd.read_class_names(root / "Training" / "obj.names.txt")
    params = params_for(profile, **overrides)
    requested = detector
    if fallback:
        det_impl, fallback_note = dets.resolve(detector, params)
    else:
        det_impl, fallback_note = DETECTORS[detector](params), None
    detector = det_impl.name
    if fallback_note and not quiet:
        print(f"NOTE: {fallback_note}", file=sys.stderr)
    pairs = rd.list_split(root, split)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit(f"No images found for split {split!r} under {root}")

    per_thr = {t: dict(tp=0, n_pred=0, tp_by_class=Counter(), centre_err=[], fp_empty=0, class_ok=0, class_n=0)
               for t in IOU_THRESHOLDS}
    gt_total, gt_by_class, degenerate_gt = 0, Counter(), 0
    images_with_gt, images_found = 0, 0
    times_ms = []
    packets = dict(produced=0, exactly_64=0, crc_valid=0, pack_errors=0)
    size_rows = []
    sizes_seen = Counter()

    for k, (img_path, lbl_path) in enumerate(pairs, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read {img_path}")
        h, w = img.shape[:2]
        sizes_seen[f"{w}x{h}"] += 1
        gts = rd.parse_yolo_labels(lbl_path.read_text() if lbl_path.exists() else "", w, h)
        gt_total += len(gts)
        images_with_gt += bool(gts)
        for g in gts:
            gt_by_class[names[g.cls] if 0 <= g.cls < len(names) else str(g.cls)] += 1
            degenerate_gt += g.w < 1 or g.h < 1

        # ---- the pipeline under test: detector -> 64 bytes -> decode from bytes only
        t0 = time.perf_counter()
        decoded_all = []
        for fields in det_impl.detect(img, gts):
            raw_packet, decoded = None, None
            try:
                raw_packet = pk.pack(pk.Detection(**dets.packet_fields(fields), **MISSION))
            except pk.PacketError:
                packets["pack_errors"] += 1
                continue
            try:
                decoded = pk.unpack(raw_packet)
            except pk.PacketError:
                decoded = None
            packets["produced"] += 1
            packets["exactly_64"] += len(raw_packet) == pk.PACKET_SIZE
            packets["crc_valid"] += decoded is not None
            if decoded is not None:
                decoded_all.append(decoded)
        times_ms.append((time.perf_counter() - t0) * 1000)

        preds = [rd.polygon_bbox(rd.decoded_polygon_px(d, params.meters_per_pixel, w, h)) for d in decoded_all]
        centres = [rd.decoded_centre_px(d, params.meters_per_pixel, w, h) for d in decoded_all]
        images_found += bool(decoded_all)

        size_matched = set()
        for t in IOU_THRESHOLDS:
            s = per_thr[t]
            s["n_pred"] += len(preds)
            if not gts:
                s["fp_empty"] += len(preds)
            for i, j, _iou in rd.match(preds, gts, t):
                s["tp"] += 1
                g = gts[j]
                s["tp_by_class"][names[g.cls] if 0 <= g.cls < len(names) else str(g.cls)] += 1
                gx, gy = g.centre
                pred_cls = decoded_all[i].class_id - dets.DATASET_CLASS_OFFSET
                if pred_cls in (0, 1) and not det_impl.uses_ground_truth:   # detector names MILCO/NOMBO
                    s["class_n"] += 1
                    s["class_ok"] += pred_cls == g.cls
                s["centre_err"].append(float(np.hypot(centres[i][0] - gx, centres[i][1] - gy)))
                if t == SIZE_MATCH_IOU:
                    size_matched.add(j)

        if size_matched:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            full_bytes = img_path.stat().st_size
            for j in size_matched:
                crop = rd.padded_crop(grey, gts[j], CROP_PAD)
                size_rows.append(dict(full=full_bytes, **{f"q{q}": rd.jpeg_size(crop, q) for q in JPEG_QUALITIES}))

        if not quiet and (k % 100 == 0 or k == len(pairs)):
            print(f"  {k}/{len(pairs)} images", file=sys.stderr)

    n_img = len(pairs)
    # At most max_per_image detections per image caps recall (one packet per image for classical).
    upper_bound = images_with_gt if det_impl.max_per_image == 1 else gt_total
    detection = {}
    for t in IOU_THRESHOLDS:
        s = per_thr[t]
        precision, recall = rd.precision_recall(s["tp"], s["n_pred"], gt_total)
        fp = s["n_pred"] - s["tp"]
        detection[f"iou_{t}"] = dict(
            iou_threshold=t,
            true_positives=s["tp"],
            predictions=s["n_pred"],
            ground_truth=gt_total,
            precision_pct=pct(precision),
            recall_pct=pct(recall),
            false_alarms=fp,
            false_alarms_per_image=round(fp / n_img, 3),
            false_alarms_on_images_without_objects=s["fp_empty"],
            false_alarms_on_images_with_objects=fp - s["fp_empty"],
            recall_by_class={c: dict(tp=s["tp_by_class"][c], gt=gt_by_class[c],
                                     recall_pct=pct(s["tp_by_class"][c] / gt_by_class[c]) if gt_by_class[c] else None)
                             for c in names},
            class_agreement_of_matches_pct=pct(s["class_ok"] / s["class_n"]) if s["class_n"] else None,
            centre_error_px_median=median_or_none(s["centre_err"]),
            centre_error_px_p95=percentile_or_none(s["centre_err"], 95),
        )

    def size_block(key):
        vals = [r[key] for r in size_rows]
        if not vals:
            return None
        med = float(statistics.median(vals))
        return dict(median_bytes=round(med, 1), hexadepth_reduction_pct=round(100 * (1 - pk.PACKET_SIZE / med), 2),
                    send_seconds_at_100bps=round(rd.send_seconds(med, LINK_BPS), 2))

    report = dict(
        meta=dict(
            dataset=rd.DATASET_NAME,
            citation=rd.DATASET_CITATION,
            doi=rd.DATASET_DOI,
            data_doi=rd.DATASET_DATA_DOI,
            split=split,
            detector=detector,
            detector_requested=requested,
            detector_label=det_impl.label,
            detector_fallback_note=fallback_note,
            detector_credit=getattr(det_impl, "credit", None),
            detector_settings=detector_settings(det_impl),
            training_overlap=training_overlap(detector, split),
            detector_uses_ground_truth=det_impl.uses_ground_truth,
            split_years=list(rd.SPLITS[split]),
            **rd.split_scheme(split),
            profile=profile,
            overrides=overrides,
            vision_params=dataclasses.asdict(params),
            limit=limit,
            images=n_img,
            image_sizes=dict(sizes_seen),
            images_with_objects=images_with_gt,
            images_without_objects=n_img - images_with_gt,
            ground_truth_objects=gt_total,
            ground_truth_by_class=dict(gt_by_class),
            degenerate_ground_truth_boxes=degenerate_gt,
            class_names={i: n for i, n in enumerate(names)},
            date=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            cpu=cpu_model(),
            cpu_threads=os.cpu_count(),
            python=platform.python_version(),
            opencv=cv2.__version__,
            notes=([
                "ORACLE MODE: ground-truth boxes are used as detections. These numbers measure only "
                "what the 64-byte packet preserves (centre, size, outline); they are NOT detection results.",
            ] if det_impl.uses_ground_truth else [
                "One packet per image: the classical detector reports at most one object per image, so "
                "recall cannot exceed recall_upper_bound_pct.",
            ] if det_impl.max_per_image == 1 else [
                "Several detections per image allowed: every box above the confidence threshold becomes "
                "its own 64-byte packet. The outline is extracted inside the box and fitted to it.",
            ]) + ([f"Authors' YOLOv4 numbers {dets.AUTHORS_TRAINING_NOTE}."] if detector == "authors_yolov4" else [])
            + ([f"FALLBACK: {fallback_note}"] if fallback_note else []) + [
                "Box IoU: bounding box of the polygon rebuilt from the decoded 64 bytes vs the "
                "ground-truth YOLO box. The dataset has no pixel masks.",
                "Ground-truth boxes with zero width or height are kept and count as misses.",
                "Pixel positions are decoded with the same nominal metres-per-pixel used to encode; "
                "pixel metrics do not depend on its true value.",
                f"Size comparison uses objects matched at IoU >= {SIZE_MATCH_IOU}; crops are the "
                f"ground-truth box + {int(CROP_PAD * 100)}% padding, encoded as greyscale JPEG "
                "(smaller than colour, so the comparison is the least favourable to HexaDepth).",
                "Speed: detector + pack + unpack per image, single process, excludes file reading.",
                "Class agreement: share of matched objects whose decoded class (MILCO/NOMBO) equals the "
                "label; only for detectors that predict the dataset classes.",
            ],
        ),
        detection=detection,
        recall_upper_bound_pct=pct(upper_bound / gt_total) if gt_total else None,
        images_with_detection=images_found,
        speed=dict(per_image_ms_median=median_or_none(times_ms), per_image_ms_p95=percentile_or_none(times_ms, 95),
                   per_image_ms_mean=round(float(np.mean(times_ms)), 2)),
        packets=dict(
            produced=packets["produced"],
            pack_errors=packets["pack_errors"],
            exactly_64_bytes_pct=pct(packets["exactly_64"] / packets["produced"]) if packets["produced"] else None,
            crc_valid_pct=pct(packets["crc_valid"] / packets["produced"]) if packets["produced"] else None,
        ),
        size=dict(
            matched_objects=len(size_rows),
            matched_at_iou=SIZE_MATCH_IOU,
            packet_bytes=pk.PACKET_SIZE,
            packet_send_seconds_at_100bps=rd.send_seconds(pk.PACKET_SIZE, LINK_BPS),
            jpeg_crop_q75=size_block("q75"),
            jpeg_crop_q50=size_block("q50"),
            full_image_jpg=size_block("full"),
        ),
    )
    return report


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def to_markdown(r: dict) -> str:
    m = r["meta"]
    lines = [
        f"# HexaDepth on real side-scan sonar: {m['split']} split, `{m['detector']}` detector"
        + (f", `{m['profile']}` profile" if not m["detector_uses_ground_truth"] else ""),
        "",
        *([("> **Oracle mode:** ground-truth boxes are used as detections. These numbers show what the "
            "64-byte packet preserves, not how well objects are found."), ""] if m["detector_uses_ground_truth"] else []),
        *([f"> **Caution:** these numbers {m['training_overlap']}.", ""] if m.get("detector") == "authors_yolov4" else []),
        *([f"> **Fallback:** {m['detector_fallback_note']}", ""] if m.get("detector_fallback_note") else []),
        f"**Detector:** {m.get('detector_label', m['detector'])}"
        + (f" · {m['detector_credit']}" if m.get("detector_credit") else "")
        + (f" · settings {json.dumps(m['detector_settings'])}" if m.get("detector_settings") else ""),
        "",
        f"**Dataset:** {m['dataset']}. {m['citation']} {m['doi']} (data: {m['data_doi']})",
        "",
        (f"**Split (Batch D2):** train/val pool = {', '.join(m['pool_years'])} (val = stratified "
         f"{round(m['val_fraction'] * 100)}% slice, seed {m['val_seed']}); test = {', '.join(m['test_years'])}. "
         if m.get("scheme") == "d2" else
         f"**Split:** tune = {', '.join(m['tune_years'])}; test = {', '.join(m['test_years'])}. ")
        + f"This run: **{m['split']}** ({', '.join(m['split_years'])})"
        + (f", limited to the first {m['limit']} images" if m["limit"] else "") + ".",
        "",
        f"**Images:** {m['images']} ({m['images_with_objects']} with objects, {m['images_without_objects']} without) · "
        f"sizes {m['image_sizes']} · ground-truth objects {m['ground_truth_objects']} {m['ground_truth_by_class']} · "
        f"degenerate boxes {m['degenerate_ground_truth_boxes']}",
        "",
        f"**Date:** {m['date']} · **CPU:** {m['cpu']} ({m['cpu_threads']} threads) · Python {m['python']} · OpenCV {m['opencv']}",
        "",
        "## Detection",
        "",
        "| IoU ≥ | Recall | Precision | TP / GT | Predictions | False alarms / image | MILCO recall | NOMBO recall | Median centre error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in r["detection"].values():
        rc = d["recall_by_class"]
        cls_cells = [f"{fmt(rc[c]['recall_pct'], '%')} ({rc[c]['tp']}/{rc[c]['gt']})" for c in ("MILCO", "NOMBO") if c in rc]
        lines.append(f"| {d['iou_threshold']} | {fmt(d['recall_pct'], '%')} | {fmt(d['precision_pct'], '%')} | "
                     f"{d['true_positives']} / {d['ground_truth']} | {d['predictions']} | {d['false_alarms_per_image']} | "
                     + " | ".join(cls_cells) + f" | {fmt(d['centre_error_px_median'], ' px')} |")
    lines += [""] + [
        f"False alarms at IoU ≥ {d['iou_threshold']}: {d['false_alarms_on_images_without_objects']} on the "
        f"{m['images_without_objects']} images with no object, {d['false_alarms_on_images_with_objects']} on the "
        f"{m['images_with_objects']} images with objects." for d in r["detection"].values()]
    agree = [f"{d['class_agreement_of_matches_pct']}% at IoU ≥ {d['iou_threshold']}" for d in r["detection"].values()
             if d.get("class_agreement_of_matches_pct") is not None]
    if agree:
        lines += ["", "Class agreement (decoded MILCO/NOMBO = label) among matched objects: " + ", ".join(agree) + "."]
    lines += [
        "",
        f"Recall ceiling for this detector: {fmt(r['recall_upper_bound_pct'], '%')}. "
        f"Images with a detection: {r['images_with_detection']} of {m['images']}.",
        "",
        "## Speed",
        "",
        f"Per image (detector + pack + unpack): median {fmt(r['speed']['per_image_ms_median'], ' ms')}, "
        f"95th percentile {fmt(r['speed']['per_image_ms_p95'], ' ms')}.",
        "",
        "## Size: what would be sent for each matched object",
        "",
        f"Matched objects (IoU ≥ {r['size']['matched_at_iou']}): {r['size']['matched_objects']}",
        "",
        "| Payload | Median bytes | HexaDepth reduction | Send time at 100 bps |",
        "|---|---|---|---|",
        f"| HexaDepth packet | {r['size']['packet_bytes']} | n/a | {r['size']['packet_send_seconds_at_100bps']} s |",
    ]
    for key, label in (("jpeg_crop_q75", "JPEG crop, quality 75"), ("jpeg_crop_q50", "JPEG crop, quality 50"),
                       ("full_image_jpg", "Full original .jpg")):
        b = r["size"][key]
        if b:
            lines.append(f"| {label} | {b['median_bytes']:,} | {b['hexadepth_reduction_pct']}% | {b['send_seconds_at_100bps']} s |")
        else:
            lines.append(f"| {label} | n/a | n/a | n/a |")
    p = r["packets"]
    lines += [
        "",
        "## Packets",
        "",
        f"Produced {p['produced']} · exactly 64 bytes {fmt(p['exactly_64_bytes_pct'], '%')} · "
        f"valid CRC on decode {fmt(p['crc_valid_pct'], '%')} · pack errors {p['pack_errors']}",
        "",
        "## Vision parameters",
        "",
        "```json",
        json.dumps(m["vision_params"], indent=2),
        "```",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in m["notes"]],
        "",
    ]
    return "\n".join(lines)


def compare(root: Path, quiet: bool = False, limit: int | None = None) -> dict:
    """Every detector on the same images with identical rules (the headline comparison)."""
    runs, skipped = {}, {}
    for name in COMPARE_DETECTORS:
        if not quiet:
            print(f"== {name}", file=sys.stderr)
        try:
            runs[name] = evaluate(root, COMPARE_SPLIT, "real", limit, {}, quiet, detector=name, fallback=False)
        except dets.ModelUnavailable as e:
            skipped[name] = str(e)
            if not quiet:
                print(f"NOTE: skipping {name}: {e}", file=sys.stderr)
    if not runs:
        raise SystemExit("No detector could run")
    any_run = next(iter(runs.values()))["meta"]
    meta = {k: any_run[k] for k in ("dataset", "citation", "doi", "data_doi", "split", "split_years", "scheme",
                                   "pool_years", "val_fraction", "val_seed", "tune_years", "test_years", "limit",
                                   "images", "image_sizes", "images_with_objects", "images_without_objects",
                                   "ground_truth_objects", "ground_truth_by_class", "degenerate_ground_truth_boxes",
                                   "class_names", "cpu", "cpu_threads", "python", "opencv")}
    meta.update(
        date=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        held_out_set=("No held-out set could be built for the authors' YOLOv4: the dataset does not include "
                      "the authors' training/validation image lists. All detectors are evaluated on the "
                      "Batch D2 TEST years (2018 + 2021), which our YOLOv8n and the classical pipeline never "
                      f"saw. Authors' YOLOv4 numbers {dets.AUTHORS_TRAINING_NOTE}."),
        rules=("Same images, same code path for every detector: detector -> 64-byte packet -> decode from the "
               "bytes only -> bounding box of the decoded outline vs the labelled box; IoU 0.5 and 0.3; greedy "
               "one-to-one matching; every image counted, including images with no objects."),
        detector_order=list(runs),
    )
    return dict(meta=meta, runs=runs, skipped=skipped, headline_detector=next(iter(runs)))


def overlap_short(meta: dict) -> str:
    o = meta["training_overlap"]
    return "Unknown: may include training images" if meta["detector"] == "authors_yolov4" else \
        "No" if o == "none" else "n/a" if o.startswith("n/a") else o


def comparison_markdown(c: dict) -> str:
    m = c["meta"]
    lines = [
        "# HexaDepth on real side-scan sonar: detector comparison",
        "",
        f"> **Held-out set:** {m['held_out_set']}",
        "",
        f"**Dataset:** {m['dataset']}. {m['citation']} {m['doi']} (data: {m['data_doi']}, CC BY 4.0)",
        "",
        f"**Images:** {m['images']} from {', '.join(m['split_years'])} ({m['images_with_objects']} with objects, "
        f"{m['images_without_objects']} without) · ground-truth objects {m['ground_truth_objects']} "
        f"{m['ground_truth_by_class']} · degenerate boxes {m['degenerate_ground_truth_boxes']}"
        + (f" · limited to the first {m['limit']} images" if m["limit"] else ""),
        "",
        f"**Rules:** {m['rules']}",
        "",
        f"**Date:** {m['date']} · **CPU:** {m['cpu']} ({m['cpu_threads']} threads) · Python {m['python']} · OpenCV {m['opencv']}",
        "",
    ]
    for key in ("iou_0.5", "iou_0.3"):
        lines += [f"## Detection at IoU ≥ {key[4:]}", "",
                  "| Detector | Trained on these images? | Recall | Precision | False alarms / image | MILCO recall | NOMBO recall | Class agreement |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in c["runs"].items():
            d, rm = r["detection"][key], r["meta"]
            rc = d["recall_by_class"]
            lines.append(
                f"| {rm['detector_label']} | {overlap_short(rm)} | {fmt(d['recall_pct'], '%')} ({d['true_positives']}/{d['ground_truth']}) | "
                f"{fmt(d['precision_pct'], '%')} | {d['false_alarms_per_image']} | "
                + " | ".join(f"{fmt(rc[k]['recall_pct'], '%')} ({rc[k]['tp']}/{rc[k]['gt']})" for k in ("MILCO", "NOMBO"))
                + f" | {fmt(d.get('class_agreement_of_matches_pct'), '%')} |")
        lines.append("")
    lines += ["## CPU time per image (detector + pack + unpack)", "",
              "| Detector | Median | 95th percentile |", "|---|---|---|"]
    for r in c["runs"].values():
        sp = r["speed"]
        lines.append(f"| {r['meta']['detector_label']} | {fmt(sp['per_image_ms_median'], ' ms')} | {fmt(sp['per_image_ms_p95'], ' ms')} |")
    lines += ["", "## What gets sent per matched object (IoU ≥ 0.3)", "",
              "| Detector | Matched objects | HexaDepth packet | JPEG crop q75 (median) | JPEG crop q50 (median) | Full image (median) |",
              "|---|---|---|---|---|---|"]
    for r in c["runs"].values():
        z = r["size"]
        cell = lambda b: "n/a" if not b else f"{b['median_bytes']:,} B ({b['hexadepth_reduction_pct']}% smaller as 64 B)"
        lines.append(f"| {r['meta']['detector_label']} | {z['matched_objects']} | {z['packet_bytes']} B | "
                     f"{cell(z['jpeg_crop_q75'])} | {cell(z['jpeg_crop_q50'])} | {cell(z['full_image_jpg'])} |")
    lines += ["", "## Packets", ""]
    for r in c["runs"].values():
        pk_ = r["packets"]
        lines.append(f"- {r['meta']['detector_label']}: {pk_['produced']} packets, exactly 64 bytes "
                     f"{fmt(pk_['exactly_64_bytes_pct'], '%')}, valid CRC {fmt(pk_['crc_valid_pct'], '%')}, "
                     f"pack errors {pk_['pack_errors']}")
    if c["skipped"]:
        lines += ["", "## Not run", ""] + [f"- `{k}`: {v}" for k, v in c["skipped"].items()]
    lines += ["", "## Credits", "",
              f"- {dets.AUTHORS_CREDIT}.",
              f"- Dataset: {m['citation']} {m['doi']}; data {m['data_doi']} (CC BY 4.0).",
              "", "Per-detector details: `results/real_eval_d2-test_<detector>.md`.", ""]
    return "\n".join(lines)


def write_report(out: Path, report: dict, md: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".md").write_text(md)
    print(f"Wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true",
                    help=f"headline: run {', '.join(COMPARE_DETECTORS)} on {COMPARE_SPLIT} -> results/real_eval.json/.md")
    ap.add_argument("--split", choices=sorted(rd.SPLITS), default="test")
    ap.add_argument("--detector", choices=sorted(DETECTORS), default="classical",
                    help="authors_yolov4, yolo, classical, or oracle (ground-truth boxes; encoding ceiling)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="real")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (quick runs)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "real" / "milco")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a vision parameter (tuning experiments)")
    ap.add_argument("--out", type=Path, default=None, help="output path prefix (no extension)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not a.data.exists():
        raise SystemExit(f"Dataset not found at {a.data}. See README for download instructions.")
    if a.compare:
        c = compare(a.data, a.quiet, a.limit)
        suffix = f"_limit{a.limit}" if a.limit else ""
        out = a.out or ROOT / "results" / f"real_eval{suffix}"
        for name, r in c["runs"].items():
            write_report(out.parent / f"real_eval_{COMPARE_SPLIT}_{name}{suffix}", r, to_markdown(r))
        md = comparison_markdown(c)
        write_report(out, c, md)
        print(md)
        return
    overrides = {k: coerce(v) for k, v in (s.split("=", 1) for s in a.set)}
    report = evaluate(a.data, a.split, a.profile, a.limit, overrides, a.quiet, detector=a.detector)

    if a.out is None:
        det = report["meta"]["detector"]
        tag = det if report["meta"]["detector_uses_ground_truth"] else f"{det}_{a.profile}"
        a.out = ROOT / "results" / (f"real_eval_{a.split}_{tag}" + (f"_limit{a.limit}" if a.limit else "")
                                    + ("_custom" if overrides else ""))
    md = to_markdown(report)
    write_report(a.out, report, md)
    print(md)


if __name__ == "__main__":
    main()
