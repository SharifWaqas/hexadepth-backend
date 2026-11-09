"""Batch D2 preparation: split, YOLO dataset builder, Colab notebook, bundled real samples."""

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from hexadepth import realdata as rd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_yolo_dataset as myd  # noqa: E402


# ---------------------------------------------------------------- split
def test_d2_split_years():
    pool, te = set(rd.D2_POOL_YEARS), set(rd.D2_TEST_YEARS)
    assert pool == {"2010", "2015", "2017"} and te == {"2018", "2021"} and not pool & te
    assert set(rd.SPLITS["d2-train"]) == set(rd.SPLITS["d2-val"]) == pool
    assert pool | te == set(rd.TUNE_YEARS) | set(rd.TEST_YEARS)


def test_image_stratum():
    assert rd.image_stratum("") == "empty"
    assert rd.image_stratum("0 0.5 0.5 0.1 0.1\n") == "milco"
    assert rd.image_stratum("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n") == "nombo"


def test_batch_d_split_unchanged():
    assert rd.SPLITS["tune"] == ("2015",) and rd.SPLITS["test"] == ("2010", "2017", "2018", "2021")


def test_split_scheme_reports_years():
    d2 = rd.split_scheme("d2-test")
    assert d2["pool_years"] == ["2010", "2015", "2017"] and d2["test_years"] == ["2018", "2021"]
    assert d2["val_fraction"] == 0.15 and d2["val_seed"] == 0
    assert rd.split_scheme("test")["tune_years"] == ["2015"]


# ---------------------------------------------------------------- label cleaning
def test_clean_label_lines():
    kept, stats = myd.clean_label_lines("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0 0.1\n\n1 0.3 0.3 0.2 0.2\r\n", 2)
    assert len(kept) == 2 and stats["dropped_zero_size"] == 1 and stats["class_0"] == 1 and stats["class_1"] == 1
    assert myd.clean_label_lines("", 2) == ([], {})
    with pytest.raises(ValueError):
        myd.clean_label_lines("2 0.5 0.5 0.1 0.1", 2)
    with pytest.raises(ValueError):
        myd.clean_label_lines("0 0.5 0.5 0.1", 2)


# ---------------------------------------------------------------- dataset builder
LABELS = {"milco": "0 0.5 0.5 0.2 0.2\n", "nombo": "1 0.4 0.4 0.2 0.2\n0 0.7 0.7 0.1 0.1\n", "empty": ""}


def _fake_milco(root: Path, per_stratum: int = 7):
    """Pool years: per_stratum images of each type; test years: 2 images each."""
    (root / "Training").mkdir(parents=True)
    (root / "Training" / "obj.names.txt").write_bytes(b"MILCO\r\nNOMBO")
    _, buf = cv2.imencode(".jpg", np.full((32, 32, 3), 100, np.uint8))
    for year in ("2010", "2015", "2017", "2018", "2021"):
        d = root / year / year
        d.mkdir(parents=True)
        kinds = [k for k in LABELS for _ in range(per_stratum)] if year in rd.D2_POOL_YEARS else ["milco", "empty"]
        for n, kind in enumerate(kinds, 1):
            (d / f"{n:04d}_{year}.jpg").write_bytes(buf.tobytes())
            (d / f"{n:04d}_{year}.txt").write_text(LABELS[kind])
    # one zero-size label (unlearnable) in the pool
    (root / "2010" / "2010" / "0001_2010.txt").write_text("1 0.5 0.5 0 0.3\n")


def test_build_never_includes_test_years_and_keeps_negatives(tmp_path):
    src, out = tmp_path / "milco", tmp_path / "build" / "yolo_milco"
    _fake_milco(src)
    before = sorted((p.relative_to(src), p.stat().st_size, p.stat().st_mtime) for p in src.rglob("*") if p.is_file())
    m = myd.build(src, out, out.with_suffix(".zip"))
    after = sorted((p.relative_to(src), p.stat().st_size, p.stat().st_mtime) for p in src.rglob("*") if p.is_file())
    assert before == after                                            # source untouched

    all_files = [p.name for p in out.rglob("*") if p.is_file()]
    assert not [f for f in all_files if "2018" in f or "2021" in f]
    train = {p.name for p in (out / "images" / "train").iterdir()}
    val = {p.name for p in (out / "images" / "val").iterdir()}
    assert not train & val and len(train) + len(val) == 3 * 3 * 7
    # 7 per (year, type) stratum -> round(7 * 0.15) = 1 each -> 9 val images, every stratum present
    assert len(val) == 9
    strata = {(n[5:9], rd.image_stratum((src / n[5:9] / n[5:9] / n.replace(".jpg", ".txt")).read_text())) for n in val}
    assert strata == {(y, k) for y in rd.D2_POOL_YEARS for k in LABELS}
    s = m["splits"]
    assert s["val"]["objects"]["MILCO"] > 0 and s["val"]["objects"]["NOMBO"] > 0 and s["val"]["negative_images"] > 0
    assert s["train"]["dropped_zero_size_boxes"] + s["val"]["dropped_zero_size_boxes"] == 1
    some = sorted(train)[0]
    assert (out / "images" / "train" / some).read_bytes() == (src / some[5:9] / some[5:9] / some).read_bytes()
    assert myd.build(src, tmp_path / "again", None)["splits"]["val"]["files"] == s["val"]["files"]   # fixed seed

    yaml = (out / "data.yaml").read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml
    assert "0: MILCO" in yaml and "1: NOMBO" in yaml and "test:" not in yaml
    with zipfile.ZipFile(out.with_suffix(".zip")) as z:
        names = z.namelist()
    assert "yolo_milco/data.yaml" in names and "yolo_milco/ATTRIBUTION.md" in names
    assert not [n for n in names if "2018" in n or "2021" in n]


def test_build_rejects_unexpected_class_names(tmp_path):
    src = tmp_path / "milco"
    _fake_milco(src)
    (src / "Training" / "obj.names.txt").write_text("NOMBO\nMILCO\n")
    with pytest.raises(SystemExit):
        myd.build(src, tmp_path / "out", None)


# ---------------------------------------------------------------- Colab notebook
def test_colab_notebook_has_required_settings():
    nb = json.loads((ROOT / "notebooks" / "train_yolo_colab.ipynb").read_text())
    assert nb["cells"][0]["cell_type"] == "markdown"
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ('pip install -q "ultralytics', 'files.upload()', 'unzip', 'YOLO("yolov8n.pt")',
                   'imgsz=1024', 'epochs=80', 'patience=20', 'seed=SEED', 'SEED = 0',
                   '.val(', 'runs/detect/train/weights/best.pt', 'files.download'):
        assert needle in src, needle


# ---------------------------------------------------------------- bundled real samples (CC BY 4.0)
def test_bundled_real_samples_work_without_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "REAL_DATA", tmp_path / "no_dataset")
    client = TestClient(server.app)
    real = [s for s in client.get("/api/samples").json() if s["group"] == "real"]
    assert len(real) == 4 and all(s["bundled"] for s in real)
    a = real[0]["attribution"]
    assert a["doi"] == "https://doi.org/10.6084/m9.figshare.24574879"
    assert a["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "Santos" in a["authors"] and "Neto" in a["authors"] and a["year"] == 2024
    assert "outlines added by HexaDepth" in a["changes"]
    img = client.get(real[0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    r = client.post("/api/analyze", data={"sample": real[0]["name"]})
    assert r.status_code == 200 and r.json()["profile"] == "real"


def test_bundled_samples_are_small_and_credited():
    folder = ROOT / "web" / "samples" / "real"
    jpgs = sorted(folder.glob("*.jpg"))
    assert len(jpgs) == 4
    for p in jpgs:
        im = cv2.imread(str(p))
        assert im is not None and im.shape[1] <= 1024
    readme = (folder / "README.md").read_text()
    for needle in ("Santos", "Moura", "Torgal", "Lobo", "Neto", "2024",
                   "Side-scan sonar imaging data of underwater vehicles for mine detection",
                   "https://doi.org/10.6084/m9.figshare.24574879",
                   "https://creativecommons.org/licenses/by/4.0/", "outlines added by HexaDepth"):
        assert needle in readme, needle
    html = (ROOT / "web" / "index.html").read_text()
    assert html.count("https://doi.org/10.6084/m9.figshare.24574879") >= 1   # attribution remains accessible in simpler UI
    assert "Detector: YOLOv4 trained by Santos et al. (2024), CC BY 4.0" in html.replace("<strong>Detector:</strong>", "Detector:")
    assert html.count("https://creativecommons.org/licenses/by/4.0/") >= 1
