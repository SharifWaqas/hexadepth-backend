"""
Build an Ultralytics YOLO dataset from the MILCO side-scan sonar data (Batch D2 split).

    python scripts/make_yolo_dataset.py
    python scripts/make_yolo_dataset.py --data data/real/milco --out build/yolo_milco

TRAIN and VAL come from the 2010 + 2015 + 2017 surveys: VAL is a stratified ~15% random slice
(by survey year x image type: contains NOMBO / MILCO only / empty; fixed seed), TRAIN is the
rest. The TEST years (2018, 2021) are NEVER copied, so a model trained on this dataset has not
seen the test images.

Source files under data/real/ are only read, never modified. Images are copied byte-for-byte
(no re-encoding). Images with empty label files are kept as negatives (background images).
Label lines with zero width or height cannot be learned and are dropped; the count is reported.

Output:
    build/yolo_milco/images/{train,val}/*.jpg
    build/yolo_milco/labels/{train,val}/*.txt
    build/yolo_milco/data.yaml          (names 0: MILCO, 1: NOMBO)
    build/yolo_milco/manifest.json      (counts + exact source list)
    build/yolo_milco/ATTRIBUTION.md     (CC BY 4.0 credit)
    build/yolo_milco.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hexadepth import realdata as rd  # noqa: E402

SPLIT_MAP = {"train": "d2-train", "val": "d2-val"}   # never "d2-test"
EXPECTED_NAMES = ["MILCO", "NOMBO"]

ATTRIBUTION = """# Attribution

Images and labels: Santos, N. P., Moura, R., Torgal, G. S., Lobo, V., & Neto, M. D. C. (2024).
*Side-scan sonar imaging data of underwater vehicles for mine detection* [dataset].
https://doi.org/10.6084/m9.figshare.24574879

Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Changes: the 2010, 2015 and 2017 surveys reorganised into the Ultralytics YOLO folder layout
(train / stratified 15% val slice); label lines with zero width or height removed. Images are
unmodified.
"""


def clean_label_lines(text: str, n_classes: int) -> tuple[list[str], Counter]:
    """Keep valid YOLO lines; drop zero-size boxes. Raises on malformed lines or unknown classes."""
    kept, stats = [], Counter()
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"line {line_no}: expected 5 fields: {line!r}")
        cls = int(float(parts[0]))
        if not 0 <= cls < n_classes:
            raise ValueError(f"line {line_no}: class {cls} not in 0..{n_classes - 1}")
        xc, yc, w, h = (float(v) for v in parts[1:])
        if w <= 0 or h <= 0:
            stats["dropped_zero_size"] += 1
            continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and w <= 1 and h <= 1):
            stats["out_of_range"] += 1   # kept: Ultralytics clips; counted for transparency
        kept.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stats[f"class_{cls}"] += 1
    return kept, stats


def build(data: Path, out: Path, zip_path: Path | None) -> dict:
    names = rd.read_class_names(data / "Training" / "obj.names.txt")
    if names != EXPECTED_NAMES:
        raise SystemExit(f"Unexpected class names {names}, expected {EXPECTED_NAMES}")
    for yolo_split, split in SPLIT_MAP.items():
        if set(rd.SPLITS[split]) & set(rd.D2_TEST_YEARS):
            raise SystemExit(f"Refusing to build: {yolo_split} would include TEST years")

    if out.exists():
        shutil.rmtree(out)  # only ever our own build folder
    manifest = dict(source=str(data), pool_years=list(rd.D2_POOL_YEARS), val_fraction=rd.D2_VAL_FRACTION,
                    val_seed=rd.D2_SEED, test_years_excluded=list(rd.D2_TEST_YEARS), splits={})
    for yolo_split, split in SPLIT_MAP.items():
        img_dir, lbl_dir = out / "images" / yolo_split, out / "labels" / yolo_split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        totals, files = Counter(), []
        for img, lbl in rd.list_split(data, split):
            if img.stem.split("_")[-1] in rd.D2_TEST_YEARS:
                raise SystemExit(f"TEST image {img} reached the {yolo_split} split")
            kept, stats = clean_label_lines(lbl.read_text() if lbl.exists() else "", len(names))
            shutil.copyfile(img, img_dir / img.name)          # read-only use of the source
            (lbl_dir / f"{img.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
            totals.update(stats)
            totals["images"] += 1
            totals["negative_images"] += not kept
            files.append(str(img.relative_to(data)))
        manifest["splits"][yolo_split] = dict(
            years=list(rd.SPLITS[split]),
            images=totals["images"],
            negative_images=totals["negative_images"],
            images_by_year={y: sum(f.startswith(y) for f in files) for y in rd.SPLITS[split]},
            objects={n: totals[f"class_{i}"] for i, n in enumerate(names)},
            dropped_zero_size_boxes=totals["dropped_zero_size"],
            out_of_range_boxes=totals["out_of_range"],
            files=files,
        )

    (out / "data.yaml").write_text(
        "# MILCO side-scan sonar, Batch D2 split: train/val from 2010+2015+2017 (val = stratified\n"
        f"# {round(rd.D2_VAL_FRACTION * 100)}% slice, seed {rd.D2_SEED}). TEST years (2018, 2021) are NOT included.\n"
        "# On Colab the notebook rewrites `path` to the unzipped folder.\n"
        f"path: {out.name}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION)

    if zip_path:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    # JPEGs are already compressed; store them to save time.
                    z.write(f, f.relative_to(out.parent),
                            compress_type=zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED)
        manifest["zip_bytes"] = zip_path.stat().st_size
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "real" / "milco")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "yolo_milco")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args(argv)
    if not a.data.exists():
        raise SystemExit(f"Dataset not found at {a.data}")
    zip_path = None if a.no_zip else a.out.with_suffix(".zip")
    m = build(a.data, a.out, zip_path)
    for name, s in m["splits"].items():
        print(f"{name:5s} images {s['images']:4d} by year {s['images_by_year']} "
              f"(negatives {s['negative_images']:3d})  objects {s['objects']}  "
              f"dropped zero-size {s['dropped_zero_size_boxes']}  out-of-range {s['out_of_range_boxes']}")
    print(f"TEST years excluded: {', '.join(m['test_years_excluded'])}")
    if zip_path:
        print(f"Wrote {zip_path} ({m['zip_bytes']:,} bytes = {m['zip_bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
