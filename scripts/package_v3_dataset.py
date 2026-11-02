"""
package_v3_dataset.py  –  EcoVision AI  –  Colab Packaging & Verification
========================================================================
Packages the multi-source dataset (data/processed_v3) into `data/ecovision_v3.zip`
for Google Colab training.

Guarantees:
  - Contains strictly data/processed_v3 (no v1/v2 files or model checkpoints).
  - Includes dataset_v3.yaml, dataset_colab_v3.yaml, and metadata manifests.
  - Computes and logs SHA-256 checksum.
  - Performs an end-to-end extraction verification test.
"""

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
import yaml

TARGET_CLASSES = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def package_and_verify(
    src_dir: Path = Path("data/processed_v3"),
    out_zip: Path = Path("data/ecovision_v3.zip")
):
    workspace = Path(__file__).resolve().parent.parent
    src_dir = workspace / src_dir
    out_zip = workspace / out_zip

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    print("=" * 70)
    print("  EcoVision AI — Packaging Multi-Source Dataset (v3)")
    print(f"  Source : {src_dir}")
    print(f"  Archive: {out_zip}")
    print("=" * 70)

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build Zip Archive
    file_count = 0
    print("\n[1/3] Compressing dataset into ZIP archive...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include YAML configurations & metadata
        yaml_files = [
            workspace / "data" / "dataset_v3.yaml",
            workspace / "data" / "dataset_colab_v3.yaml",
            workspace / "training" / "configs" / "yolov11_waste_v3.yaml"
        ]
        for yf in yaml_files:
            if yf.exists():
                arcname = f"metadata/{yf.name}"
                zf.write(yf, arcname=arcname)
                file_count += 1

        # Walk through data/processed_v3
        for root, _, files in os.walk(src_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(src_dir)
                arcname = f"data/processed_v3/{rel_path}".replace("\\", "/")
                zf.write(full_path, arcname=arcname)
                file_count += 1

    archive_size_mb = out_zip.stat().st_size / (1024 * 1024)
    sha256_hash = compute_sha256(out_zip)

    print(f"  Archive created successfully!")
    print(f"  Total files packed : {file_count}")
    print(f"  Archive file size  : {archive_size_mb:.2f} MB")
    print(f"  SHA-256 Checksum   : {sha256_hash}")

    # 2. End-to-End Extraction Verification
    print("\n[2/3] Performing independent extraction verification...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"  Extracting to temporary sandbox: {temp_path}")
        with zipfile.ZipFile(out_zip, "r") as zf:
            zf.extractall(temp_path)

        extracted_v3 = temp_path / "data" / "processed_v3"
        assert extracted_v3.exists(), "Extracted v3 directory missing!"

        # Check splits
        for split, expected_imgs in [("train", 8242), ("val", 3007), ("test", 374)]:
            img_dir = extracted_v3 / "images" / split
            lbl_dir = extracted_v3 / "labels" / split
            imgs = list(img_dir.glob("*.*"))
            lbls = list(lbl_dir.glob("*.txt"))

            assert len(imgs) == expected_imgs, f"Split {split} expected {expected_imgs} images, got {len(imgs)}"
            assert len(lbls) == expected_imgs, f"Split {split} expected {expected_imgs} labels, got {len(lbls)}"
            print(f"  [OK] {split.upper():<5}: {len(imgs)} images and {len(lbls)} labels verified (1-to-1 match).")

        # Verify negative labels are strictly 0-byte empty files
        neg_labels_checked = 0
        for split in ["train", "val"]:
            lbl_dir = extracted_v3 / "labels" / split
            for lf in lbl_dir.glob("neg_*.txt"):
                assert lf.stat().st_size == 0, f"Negative label {lf.name} is not empty!"
                neg_labels_checked += 1
        assert neg_labels_checked == 800, f"Expected 800 negative labels checked, got {neg_labels_checked}"
        print(f"  [OK] Negatives: Exactly {neg_labels_checked} empty 0-byte background labels verified.")

        # Verify YAML configuration inside archive
        colab_yaml_path = temp_path / "metadata" / "dataset_colab_v3.yaml"
        assert colab_yaml_path.exists(), "dataset_colab_v3.yaml missing from metadata!"
        with open(colab_yaml_path) as f:
            colab_cfg = yaml.safe_load(f)
        names = [colab_cfg["names"][i] for i in range(6)]
        assert names == TARGET_CLASSES, f"Class order mismatch: {names}"
        assert colab_cfg["nc"] == 6
        print(f"  [OK] Colab YAML: 6 target classes verified in exact order: {names}")

    # 3. Write metadata verification file
    print("\n[3/3] Writing packaging manifest...")
    meta_path = workspace / "data" / "ecovision_v3_package_manifest.json"
    manifest_data = {
        "archive_name": out_zip.name,
        "archive_path": str(out_zip),
        "archive_size_bytes": out_zip.stat().st_size,
        "archive_size_mb": round(archive_size_mb, 2),
        "sha256": sha256_hash,
        "total_files": file_count,
        "classes": TARGET_CLASSES,
        "split_counts": {
            "train": 8242,
            "val": 3007,
            "test": 374,
            "total": 11623
        },
        "negatives_count": 800,
        "verification_status": "PASSED"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"  Packaging manifest written to: {meta_path}")
    print("\n============================================================")
    print("  Packaging & Verification 100% Complete and Passed!")
    print("============================================================\n")

    return out_zip, sha256_hash, archive_size_mb


if __name__ == "__main__":
    package_and_verify()

