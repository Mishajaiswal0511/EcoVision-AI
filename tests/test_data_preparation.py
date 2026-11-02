"""
Tests for EcoVision AI dataset preparation scripts.
Run with: pytest tests/ -v

These tests use synthetic fixtures (no real dataset download required)
to validate the logic of all preparation scripts.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Temporary directory that is cleaned up after each test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_dataset(temp_dir):
    """
    Create a minimal synthetic YOLO-format dataset with 3 classes and 12 images.
    Layout:
        temp_dir/
            images/train/  (6 images)
            images/val/    (3 images)
            images/test/   (3 images)
            labels/train/
            labels/val/
            labels/test/
            data.yaml
    """
    from PIL import Image
    import numpy as np

    class_names = ["plastic", "paper", "metal"]
    n_classes = len(class_names)

    # Write dataset.yaml
    yaml_content = (
        f"path: {temp_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {n_classes}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names))
    )
    (temp_dir / "data.yaml").write_text(yaml_content)

    splits = {"train": 6, "val": 3, "test": 3}
    for split, count in splits.items():
        img_dir = temp_dir / "images" / split
        lbl_dir = temp_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(count):
            # Create a minimal valid image (100x100 white pixel)
            img = Image.fromarray(
                (np.ones((100, 100, 3), dtype=np.uint8) * 200)
            )
            img_path = img_dir / f"img_{split}_{i:03d}.jpg"
            img.save(img_path)

            # Create a valid YOLO label: class cx cy w h
            cls_id = i % n_classes
            lbl_path = lbl_dir / f"img_{split}_{i:03d}.txt"
            lbl_path.write_text(f"{cls_id} 0.5 0.5 0.3 0.4\n")

    return temp_dir, class_names


# ---------------------------------------------------------------------------
# Tests: audit_dataset.py helpers
# ---------------------------------------------------------------------------

class TestParseLabel:
    def test_valid_label(self):
        from audit_dataset import parse_label_file
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("0 0.5 0.5 0.3 0.4\n")
        tmp.write("1 0.2 0.3 0.1 0.2\n")
        tmp.close()

        anns = parse_label_file(Path(tmp.name), n_classes=3)
        assert len(anns) == 2
        assert anns[0] == (0, 0.5, 0.5, 0.3, 0.4)
        assert anns[1] == (1, 0.2, 0.3, 0.1, 0.2)
        os.unlink(tmp.name)

    def test_invalid_class_id(self):
        from audit_dataset import parse_label_file
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("99 0.5 0.5 0.3 0.4\n")  # class 99 doesn't exist
        tmp.close()

        with pytest.raises(ValueError, match="class_id"):
            parse_label_file(Path(tmp.name), n_classes=3)
        os.unlink(tmp.name)

    def test_coordinate_out_of_range(self):
        from audit_dataset import parse_label_file
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("0 1.5 0.5 0.3 0.4\n")  # cx=1.5 is out of [0,1]
        tmp.close()

        with pytest.raises(ValueError, match="not in"):
            parse_label_file(Path(tmp.name), n_classes=3)
        os.unlink(tmp.name)

    def test_missing_fields(self):
        from audit_dataset import parse_label_file
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("0 0.5 0.5\n")  # only 3 fields
        tmp.close()

        with pytest.raises(ValueError, match="expected at least 5"):
            parse_label_file(Path(tmp.name), n_classes=3)
        os.unlink(tmp.name)

    def test_empty_file(self):
        from audit_dataset import parse_label_file
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("")
        tmp.close()

        anns = parse_label_file(Path(tmp.name), n_classes=3)
        assert anns == []
        os.unlink(tmp.name)

    def test_missing_label_file(self, temp_dir):
        from audit_dataset import parse_label_file
        anns = parse_label_file(temp_dir / "nonexistent.txt", n_classes=3)
        assert anns == []


# ---------------------------------------------------------------------------
# Tests: check_annotations.py helpers
# ---------------------------------------------------------------------------

class TestAnnotationChecker:
    def test_valid_annotation_passes(self, temp_dir):
        from check_annotations import check_split

        img_dir = temp_dir / "images" / "train"
        lbl_dir = temp_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create dummy image and valid label
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        result = check_split(temp_dir, "train", n_classes=3)

        assert result["image_count"] == 1
        assert result["valid_annotations"] == 1
        assert result["issue_counts"]["field_count_errors"] == 0
        assert result["issue_counts"]["out_of_range_errors"] == 0
        assert result["issue_counts"]["invalid_class_id_errors"] == 0

    def test_degenerate_box_flagged(self, temp_dir):
        from check_annotations import check_split

        img_dir = temp_dir / "images" / "train"
        lbl_dir = temp_dir / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.0 0.2\n")  # w=0 → degenerate

        result = check_split(temp_dir, "train", n_classes=3)
        assert result["issue_counts"]["degenerate_box_errors"] == 1

    def test_skipped_for_missing_split(self, temp_dir):
        from check_annotations import check_split
        result = check_split(temp_dir, "train", n_classes=3)
        assert result.get("skipped") is True


# ---------------------------------------------------------------------------
# Tests: deduplicate_dataset.py helpers
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_md5_identical_files(self, temp_dir):
        from deduplicate_dataset import md5_of_file

        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        content = b"fake image bytes"
        p1.write_bytes(content)
        p2.write_bytes(content)

        assert md5_of_file(p1) == md5_of_file(p2)

    def test_md5_different_files(self, temp_dir):
        from deduplicate_dataset import md5_of_file

        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        p1.write_bytes(b"image one")
        p2.write_bytes(b"image two")

        assert md5_of_file(p1) != md5_of_file(p2)

    def test_phash_same_image(self, temp_dir):
        from deduplicate_dataset import phash_of_image
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        img.save(p1)
        img.save(p2)

        h1 = phash_of_image(p1)
        h2 = phash_of_image(p2)
        assert h1 is not None
        assert (h1 - h2) == 0

    def test_phash_different_images(self, temp_dir):
        from deduplicate_dataset import phash_of_image
        from PIL import Image
        import numpy as np

        img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
        p1 = temp_dir / "a.jpg"
        p2 = temp_dir / "b.jpg"
        img1.save(p1)
        img2.save(p2)

        h1 = phash_of_image(p1)
        h2 = phash_of_image(p2)
        assert (h1 - h2) > 0


# ---------------------------------------------------------------------------
# Tests: prepare_splits.py helpers
# ---------------------------------------------------------------------------

class TestPrepareSplits:
    def test_get_primary_class(self, temp_dir):
        from prepare_splits import get_primary_class

        lbl = temp_dir / "test.txt"
        lbl.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.2 0.2\n")
        # Class 0 appears twice, class 1 once → primary = 0
        assert get_primary_class(lbl) == 0

    def test_split_ratios_sum(self):
        """Validate that prepare_splits rejects ratios that don't sum to 1."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/prepare_splits.py",
             "--source", "data/raw",
             "--out_dir", "data/processed",
             "--classes", "data/dataset.yaml",
             "--train", "0.6",
             "--val", "0.2",
             "--test", "0.1"],  # 0.9 total — should fail
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode != 0
        assert "1.0" in result.stdout or "1.0" in result.stderr

    def test_full_audit_on_mock_dataset(self, mock_dataset):
        """Run a full audit on the synthetic dataset and verify report."""
        data_dir, class_names = mock_dataset
        import tempfile
        out_dir = Path(tempfile.mkdtemp())

        try:
            from audit_dataset import audit, write_json_report, write_markdown_report

            report = audit(
                data_dir=str(data_dir),
                classes_yaml=str(data_dir / "data.yaml"),
                out_dir=str(out_dir),
                preview_n=2,
            )

            assert report["global_summary"]["total_images"] == 12
            assert report["global_summary"]["total_annotations"] == 12
            assert report["global_summary"]["corrupt_images"] == 0
            assert report["global_summary"]["missing_label_files"] == 0
            assert report["global_summary"]["invalid_annotation_files"] == 0

            # Check split counts
            assert report["splits"]["train"]["image_count"] == 6
            assert report["splits"]["val"]["image_count"] == 3
            assert report["splits"]["test"]["image_count"] == 3

            # Write reports and verify they exist
            write_json_report(report, out_dir / "audit_report.json")
            write_markdown_report(report, out_dir / "audit_report.md")

            assert (out_dir / "audit_report.json").exists()
            assert (out_dir / "audit_report.md").exists()

            # Verify JSON is valid
            with open(out_dir / "audit_report.json") as f:
                loaded = json.load(f)
            assert loaded["class_names"] == class_names

        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: label_mapping.json validity
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_label_mapping_json_valid(self):
        mapping_path = Path(__file__).parent.parent / "data" / "label_mapping.json"
        assert mapping_path.exists(), f"label_mapping.json not found at {mapping_path}"

        with open(mapping_path) as f:
            mapping = json.load(f)

        assert "target_classes" in mapping, "Missing 'target_classes' key"
        assert "mappings" in mapping, "Missing 'mappings' key"
        assert len(mapping["target_classes"]) > 0, "target_classes is empty"

        target_set = set(mapping["target_classes"])
        for src, tgt in mapping["mappings"].items():
            if src.startswith("_"):
                continue
            if tgt is not None:
                assert tgt in target_set, (
                    f"Source class '{src}' maps to '{tgt}' "
                    f"which is not in target_classes."
                )

    def test_dataset_yaml_valid(self):
        import yaml
        yaml_path = Path(__file__).parent.parent / "data" / "dataset.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert "nc" in cfg
        assert "names" in cfg
        assert "train" in cfg
        assert "val" in cfg
        assert "test" in cfg

    def test_source_manifest_valid(self):
        manifest_path = Path(__file__).parent.parent / "data" / "source_license_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "datasets" in manifest
        for ds in manifest["datasets"]:
            assert "id" in ds
            assert "license" in ds
            assert "annotation_type" in ds

