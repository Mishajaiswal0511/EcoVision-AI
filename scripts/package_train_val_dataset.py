import hashlib
import json
import zipfile
from pathlib import Path

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def package_train_val():
    workspace = Path(__file__).resolve().parent.parent
    src_root = workspace / "data" / "processed_v3"
    out_zip = workspace / "data" / "ecovision_train_val.zip"

    print("=" * 70)
    print("  Creating Train-Validation-Only Dataset Package")
    print(f"  Source : {src_root}")
    print(f"  Target : {out_zip}")
    print("=" * 70)

    # Dataset YAML tailored strictly for train and val
    yaml_content = """# EcoVision AI — Train and Validation Only (Test Excluded)
path: /tmp/dataset/data/processed_v3
train: images/train
val: images/val
names:
  0: plastic
  1: metal
  2: cardboard
  3: glass
  4: paper
  5: general_trash
nc: 6
"""

    train_imgs = list((src_root / "images" / "train").glob("*.*"))
    train_lbls = list((src_root / "labels" / "train").glob("*.txt"))
    val_imgs = list((src_root / "images" / "val").glob("*.*"))
    val_lbls = list((src_root / "labels" / "val").glob("*.txt"))

    print(f"Train images: {len(train_imgs):5d} | labels: {len(train_lbls):5d}")
    print(f"Val   images: {len(val_imgs):5d} | labels: {len(val_lbls):5d}")
    assert len(train_imgs) == 8242, f"Expected 8242 train images, found {len(train_imgs)}"
    assert len(train_lbls) == 8242, f"Expected 8242 train labels, found {len(train_lbls)}"
    assert len(val_imgs) == 3007, f"Expected 3007 val images, found {len(val_imgs)}"
    assert len(val_lbls) == 3007, f"Expected 3007 val labels, found {len(val_lbls)}"

    # Build zip
    print("\nCompressing into ecovision_train_val.zip...")
    total_added = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write dataset YAML
        zf.writestr("metadata/dataset_train_val.yaml", yaml_content)

        # Write train images & labels
        for f in train_imgs:
            zf.write(f, arcname=f"data/processed_v3/images/train/{f.name}")
            total_added += 1
        for f in train_lbls:
            zf.write(f, arcname=f"data/processed_v3/labels/train/{f.name}")
            total_added += 1

        # Write val images & labels
        for f in val_imgs:
            zf.write(f, arcname=f"data/processed_v3/images/val/{f.name}")
            total_added += 1
        for f in val_lbls:
            zf.write(f, arcname=f"data/processed_v3/labels/val/{f.name}")
            total_added += 1

    print(f"Total files written: {total_added} + 1 metadata YAML")
    file_size_bytes = out_zip.stat().st_size
    file_size_mb = file_size_bytes / (1024 ** 2)
    sha256 = compute_sha256(out_zip)

    print("\n[VERIFICATION]")
    print(f"Archive Size : {file_size_bytes} bytes ({file_size_mb:.2f} MB)")
    print(f"SHA-256      : {sha256}")

    # Verify no test or real_world_eval files leaked in
    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        test_files = [n for n in names if "test" in n.lower() and not n.endswith(".yaml")]
        real_world_files = [n for n in names if "real_world" in n.lower() or "ood" in n.lower()]
        
        print(f"Total files in archive : {len(names)}")
        print(f"Test files present     : {len(test_files)} (Must be 0)")
        print(f"Real-world eval present: {len(real_world_files)} (Must be 0)")

        assert len(test_files) == 0, f"SECURITY ERROR: Test files leaked into package: {test_files[:5]}"
        assert len(real_world_files) == 0, f"SECURITY ERROR: Real world eval leaked into package!"
        assert len(names) == 8242 * 2 + 3007 * 2 + 1, f"Mismatch in archive contents! Found {len(names)}"

    manifest = {
        "archive_name": out_zip.name,
        "archive_path": str(out_zip),
        "archive_size_bytes": file_size_bytes,
        "archive_size_mb": round(file_size_mb, 2),
        "sha256": sha256,
        "total_files": len(names),
        "splits": {
            "train": {"images": 8242, "labels": 8242},
            "val": {"images": 3007, "labels": 3007},
            "test": {"images": 0, "labels": 0, "status": "EXCLUDED"}
        },
        "test_split_excluded": True,
        "real_world_eval_excluded": True,
        "verification_status": "PASSED"
    }

    manifest_path = workspace / "data" / "ecovision_train_val_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to: {manifest_path}")
    print("Train-val packaging PASSED successfully!")

if __name__ == "__main__":
    package_train_val()

