import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

EXPECTED_SHA256 = "3595e44a1791fac2684ace09c8e24b3d20936a68f8fd2377217ab9140cae8d29"
DATASET_ID = "mishajaiswal0511/ecovision-train-val"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def run_phase_1():
    print("=" * 70)
    print("  PHASE 1 — DATASET PRE-UPLOAD AUDIT & PRIVATE UPLOAD")
    print("=" * 70)

    workspace = Path(__file__).resolve().parent.parent
    zip_path = workspace / "data" / "ecovision_train_val.zip"
    cli_path = workspace / ".venv" / "Scripts" / "kaggle.exe"

    # Check 1: File existence
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    print(f"[1/7] Archive located: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")

    # Check 2: SHA-256 calculation & comparison
    print("[2/7] Recalculating SHA-256 checksum...")
    actual_sha = compute_sha256(zip_path)
    print(f"      Calculated : {actual_sha}")
    print(f"      Approved   : {EXPECTED_SHA256}")
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"CRITICAL: SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nGot: {actual_sha}")
    print("      SHA-256 Checksum MATCHES EXACTLY!")

    # Check 3 & 4 & 5: File listing & exclusion checks
    print("[3/7] Inspecting archive contents...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        total_files = len(names)
        train_imgs = [n for n in names if n.startswith("data/processed_v3/images/train/")]
        train_lbls = [n for n in names if n.startswith("data/processed_v3/labels/train/")]
        val_imgs = [n for n in names if n.startswith("data/processed_v3/images/val/")]
        val_lbls = [n for n in names if n.startswith("data/processed_v3/labels/val/")]
        test_files = [n for n in names if "test" in n.lower() and not n.endswith(".yaml")]
        real_world_files = [n for n in names if "real_world" in n.lower() or "ood" in n.lower()]

        print(f"      Total files       : {total_files}")
        print(f"      Train images      : {len(train_imgs)} (Expected: 8242)")
        print(f"      Train labels      : {len(train_lbls)} (Expected: 8242)")
        print(f"      Validation images : {len(val_imgs)} (Expected: 3007)")
        print(f"      Validation labels : {len(val_lbls)} (Expected: 3007)")
        print(f"      Test files        : {len(test_files)} (Must be 0)")
        print(f"      Real-world files  : {len(real_world_files)} (Must be 0)")

        assert len(train_imgs) == 8242, f"Train images mismatch: {len(train_imgs)}"
        assert len(train_lbls) == 8242, f"Train labels mismatch: {len(train_lbls)}"
        assert len(val_imgs) == 3007, f"Val images mismatch: {len(val_imgs)}"
        assert len(val_lbls) == 3007, f"Val labels mismatch: {len(val_lbls)}"
        assert len(test_files) == 0, f"SECURITY FAILURE: Test files present! {test_files[:5]}"
        assert len(real_world_files) == 0, f"SECURITY FAILURE: Real world eval present!"
        assert "metadata/dataset_train_val.yaml" in names, "Missing dataset YAML!"
    print("      Archive content integrity VERIFIED!")

    # Check 6: Dataset metadata configuration
    print("[4/7] Verifying dataset metadata configuration...")
    upload_dir = workspace / "data" / "kaggle_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    meta_path = upload_dir / "dataset-metadata.json"

    meta_content = {
        "title": "EcoVision Train-Val Dataset v3",
        "id": DATASET_ID,
        "licenses": [
            {
                "name": "CC-BY-4.0"
            }
        ]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)
    print(f"      Metadata configured for ID: {DATASET_ID} (Default Private)")

    # Copy archive into upload_dir (or symlink/move)
    dest_zip = upload_dir / "ecovision_train_val.zip"
    if not dest_zip.exists() or dest_zip.stat().st_size != zip_path.stat().st_size:
        print(f"      Copying archive to upload staging directory {upload_dir}...")
        shutil.copy2(zip_path, dest_zip)
    print("      Staging complete.")

    # Check 7: Kaggle authentication
    print("[5/7] Verifying Kaggle authentication...")
    auth_check = subprocess.run([str(cli_path), "datasets", "list", "--mine", "--max-size", "1"], capture_output=True, text=True)
    if auth_check.returncode != 0:
        raise RuntimeError("Kaggle authentication is not active!")
    print("      Kaggle authentication is ACTIVE.")

    # Execute Upload
    print("\n[6/7] Executing Kaggle private dataset upload (kaggle datasets create)...")
    cmd = [str(cli_path), "datasets", "create", "-p", str(upload_dir)]
    print(f"      Running command: {' '.join(cmd)}")
    
    upload_res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"      Upload stdout: {upload_res.stdout.strip()}")
    if upload_res.stderr.strip():
        print(f"      Upload stderr: {upload_res.stderr.strip()}")
    
    if upload_res.returncode != 0 and "already exists" not in upload_res.stderr and "already exists" not in upload_res.stdout:
        raise RuntimeError(f"Kaggle upload failed with code {upload_res.returncode}")

    # Check 8: Verify upload status on Kaggle
    print("\n[7/7] Verifying dataset status on Kaggle...")
    for attempt in range(15):
        time.sleep(5)
        status_res = subprocess.run([str(cli_path), "datasets", "status", DATASET_ID], capture_output=True, text=True)
        status_str = status_res.stdout.strip()
        print(f"      Attempt {attempt+1}: Status = {status_str}")
        if "ready" in status_str.lower():
            print("      Dataset is READY on Kaggle!")
            break
        elif "error" in status_str.lower():
            raise RuntimeError(f"Kaggle dataset processing error: {status_str}")

    # Record report
    report = {
        "phase": "PHASE 1 — PRIVATE DATASET UPLOAD",
        "dataset_id": DATASET_ID,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_ID}",
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_size_mb": round(zip_path.stat().st_size / (1024**2), 2),
        "sha256": actual_sha,
        "visibility": "private",
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "test_images_excluded": True,
        "real_world_eval_excluded": True,
        "upload_status": "READY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = workspace / "docs" / "phase1_dataset_upload_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE — DATASET UPLOAD VERIFIED SUCCESSFULLY!")
    print(f"  URL: https://www.kaggle.com/datasets/{DATASET_ID}")
    print("=" * 70)

if __name__ == "__main__":
    run_phase_1()

