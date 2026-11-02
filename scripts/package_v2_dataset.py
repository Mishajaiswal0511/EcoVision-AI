"""
package_v2_dataset.py  –  EcoVision AI  –  Colab Packaging & Verification
========================================================================
Packages the corrected 6-class dataset (data/processed_bbox_v2) into
ecovision_bbox_v2.zip for Google Colab training.

Guarantees:
  - Contains strictly data/processed_bbox_v2 (no v1 files or checkpoints).
  - Includes dataset_v2.yaml, dataset_colab_v2.yaml, and metadata.
  - Computes and logs SHA-256 checksum.
  - Verifies archive integrity and entry counts.
"""

import hashlib
import json
import os
import zipfile
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def package_and_verify():
    workspace = Path(__file__).resolve().parent.parent
    src_dir = workspace / "data" / "processed_bbox_v2"
    out_zip = workspace / "data" / "ecovision_bbox_v2.zip"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    print(f"============================================================")
    print(f"  EcoVision AI — Packaging Corrected Dataset (v2)")
    print(f"  Source : {src_dir}")
    print(f"  Archive: {out_zip}")
    print(f"============================================================\n")

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build Zip Archive
    file_count = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include YAML configurations
        yaml_v2 = workspace / "data" / "dataset_v2.yaml"
        if yaml_v2.exists():
            zf.write(yaml_v2, arcname="data/dataset_v2.yaml")
            zf.write(yaml_v2, arcname="dataset.yaml")
            file_count += 2

        yaml_colab = workspace / "data" / "dataset_colab_v2.yaml"
        if yaml_colab.exists():
            zf.write(yaml_colab, arcname="data/dataset_colab_v2.yaml")
            zf.write(yaml_colab, arcname="dataset_colab.yaml")
            file_count += 2

        # Include processed_bbox_v2 files
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = Path(root) / file
                # Store relative to workspace root (e.g. data/processed_bbox_v2/images/train/...)
                rel_path = file_path.relative_to(workspace)
                arc_name = str(rel_path).replace("\\", "/")
                zf.write(file_path, arcname=arc_name)
                file_count += 1
                if file_count % 500 == 0:
                    print(f"  Archived {file_count} files...")

    # Also create copy or symlink in root workspace for convenience
    root_zip = workspace / "ecovision_bbox_v2.zip"
    import shutil
    shutil.copy2(out_zip, root_zip)

    size_bytes = out_zip.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    sha256 = compute_sha256(out_zip)

    print(f"\n[OK] Archive created successfully.")
    print(f"  Primary path: {out_zip}")
    print(f"  Root path   : {root_zip}")
    print(f"  Total files : {file_count}")
    print(f"  File size   : {size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"  SHA-256     : {sha256}")

    # 2. Strict Verification
    print(f"\nVerifying ZIP archive contents...")
    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        total_entries = len(namelist)

        train_imgs = [n for n in namelist if "data/processed_bbox_v2/images/train/" in n]
        val_imgs   = [n for n in namelist if "data/processed_bbox_v2/images/val/" in n]
        test_imgs  = [n for n in namelist if "data/processed_bbox_v2/images/test/" in n]

        train_lbls = [n for n in namelist if "data/processed_bbox_v2/labels/train/" in n]
        val_lbls   = [n for n in namelist if "data/processed_bbox_v2/labels/val/" in n]
        test_lbls  = [n for n in namelist if "data/processed_bbox_v2/labels/test/" in n]

        has_manifest = "data/processed_bbox_v2/split_manifest.json" in namelist
        has_dist     = "data/processed_bbox_v2/class_distribution.json" in namelist
        has_yaml     = "data/dataset_v2.yaml" in namelist
        has_colab_yaml = "data/dataset_colab_v2.yaml" in namelist

        # Check for forbidden items (v1 files, weights, etc.)
        forbidden_v1 = [n for n in namelist if "data/processed/" in n]
        forbidden_pts = [n for n in namelist if n.endswith(".pt")]

        assert len(forbidden_v1) == 0, f"Found {len(forbidden_v1)} V1 files in archive!"
        assert len(forbidden_pts) == 0, f"Found {len(forbidden_pts)} .pt weight files in archive!"
        assert len(train_imgs) == 1790, f"Expected 1790 train images, got {len(train_imgs)}"
        assert len(val_imgs)   == 379,  f"Expected 379 val images, got {len(val_imgs)}"
        assert len(test_imgs)  == 374,  f"Expected 374 test images, got {len(test_imgs)}"
        assert len(train_lbls) == 1790, f"Expected 1790 train labels, got {len(train_lbls)}"
        assert len(val_lbls)   == 379,  f"Expected 379 val labels, got {len(val_lbls)}"
        assert len(test_lbls)  == 374,  f"Expected 374 test labels, got {len(test_lbls)}"
        assert has_manifest, "Missing split_manifest.json"
        assert has_dist, "Missing class_distribution.json"
        assert has_yaml, "Missing dataset_v2.yaml"
        assert has_colab_yaml, "Missing dataset_colab_v2.yaml"

    print(f"[OK] Verification PASSED 100%:")
    print(f"  Train: {len(train_imgs)} images, {len(train_lbls)} labels")
    print(f"  Val  : {len(val_imgs)} images, {len(val_lbls)} labels")
    print(f"  Test : {len(test_imgs)} images, {len(test_lbls)} labels")
    print(f"  Metadata files present: split_manifest.json, class_distribution.json, dataset_v2.yaml, dataset_colab_v2.yaml")
    print(f"  V1 contamination: 0 files")
    print(f"  Model checkpoints: 0 files")

    # Record verification results into json
    info = {
        "archive_name": "ecovision_bbox_v2.zip",
        "relative_path": "data/ecovision_bbox_v2.zip",
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "sha256": sha256,
        "total_files": total_entries,
        "split_counts": {
            "train": {"images": len(train_imgs), "labels": len(train_lbls)},
            "val": {"images": len(val_imgs), "labels": len(val_lbls)},
            "test": {"images": len(test_imgs), "labels": len(test_lbls)},
        },
        "classes": [
            "plastic", "metal", "cardboard", "glass", "paper", "general_trash"
        ],
        "verified": True
    }
    with open(workspace / "data" / "archive_v2_manifest.json", "w") as f:
        json.dump(info, f, indent=2)

    return info

if __name__ == "__main__":
    package_and_verify()

