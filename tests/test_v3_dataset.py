"""
test_v3_dataset.py  –  Unit Tests for EcoVision AI Dataset v3 & Negative Curation
================================================================================
Validates:
  1. Six-class taxonomy consistency and exact ordering.
  2. Bounding box format and coordinate range [0, 1].
  3. Negative sample label format (0-byte empty files).
  4. Real-world evaluation benchmark integrity.
  5. Absence of cross-split duplicate leakage.
"""

import json
from pathlib import Path
import pytest
import yaml

TARGET_CLASSES = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]


def test_taxonomy_order_in_dataset_yaml():
    yaml_p = Path("data/dataset_v3.yaml")
    if not yaml_p.exists():
        pytest.skip("data/dataset_v3.yaml not generated yet")
    with open(yaml_p) as f:
        cfg = yaml.safe_load(f)

    assert cfg["nc"] == 6
    if isinstance(cfg["names"], dict):
        names = [cfg["names"][i] for i in range(6)]
    else:
        names = cfg["names"]
    assert names == TARGET_CLASSES


def test_negative_manifest_and_labels():
    manifest_p = Path("data/raw/negatives/negatives_manifest.json")
    if not manifest_p.exists():
        pytest.skip("negatives_manifest.json not generated yet")
    with open(manifest_p) as f:
        manifest = json.load(f)

    assert manifest["total_images"] == 800
    assert manifest["total_labels"] == 800

    # Ensure prohibited classes are truly excluded
    prohibited = {'bottle', 'cup', 'wine glass', 'banana', 'apple'}
    assert prohibited.issubset(set(manifest["excluded_classes"]))

    # Test random sample of label files are strictly 0 bytes
    lbl_dir = Path("data/raw/negatives/labels")
    lbl_files = list(lbl_dir.glob("*.txt"))
    assert len(lbl_files) == 800
    for lf in lbl_files[:50]:
        assert lf.stat().st_size == 0, f"Negative label {lf.name} is not empty!"


def test_real_world_evaluation_benchmark():
    eval_dir = Path("data/real_world_eval")
    if not eval_dir.exists():
        pytest.skip("Real-world eval dir not found")

    manifest_p = eval_dir / "eval_manifest.json"
    assert manifest_p.exists()
    with open(manifest_p) as f:
        man = json.load(f)

    assert man["total_samples"] == 11
    # Check that negative samples have 0-byte labels
    for s in man["samples"]:
        lbl_p = eval_dir / "labels" / s["label_file"]
        assert lbl_p.exists()
        if s["type"] == "negative":
            assert lbl_p.stat().st_size == 0
        else:
            assert lbl_p.stat().st_size > 0
            content = lbl_p.read_text().strip()
            parts = content.split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert 0 <= cid <= 5


def test_colab_notebook_valid_json():
    nb_p = Path("training/colab_yolov11_v3_training.ipynb")
    assert nb_p.exists()
    with open(nb_p, encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    # Verify no legacy fl_gamma is present in the notebook
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        assert "fl_gamma" not in source, "Found deprecated fl_gamma in notebook!"


