import json
from pathlib import Path

def audit_notebook():
    nb_path = Path("training/kaggle/ecovision_yolo11m_training.ipynb")
    meta_path = Path("training/kaggle/kernel-metadata.json")

    print("=" * 70)
    print("  PHASE 2 — KAGGLE NOTEBOOK & METADATA AUDIT")
    print("=" * 70)

    # 1. Audit Metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("[1/5] Auditing Kernel Metadata...")
    print(f"      Kernel ID        : {meta.get('id')}")
    print(f"      Kernel Type      : {meta.get('kernel_type')}")
    print(f"      GPU Enabled      : {meta.get('enable_gpu')}")
    print(f"      Internet Enabled : {meta.get('enable_internet')}")
    print(f"      Private Mode     : {meta.get('is_private')}")
    print(f"      Dataset Sources  : {meta.get('dataset_sources')}")

    assert meta.get('id') == "mishajaiswal0511/ecovision-yolo11m-training", "Kernel ID incorrect!"
    assert str(meta.get('enable_gpu')).lower() == "true", "GPU must be enabled!"
    assert str(meta.get('enable_internet')).lower() == "true", "Internet must be enabled for yolo11m.pt download!"
    assert str(meta.get('is_private')).lower() == "true", "Kernel must be private!"
    assert "mishajaiswal0511/ecovision-train-val" in meta.get('dataset_sources', []), "Dataset source missing!"

    # 2. Audit Notebook JSON & Cells
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("\n[2/5] Auditing Notebook Cells...")
    print(f"      Total Cells: {len(nb['cells'])}")
    all_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

    # 3. Check Hyperparameters
    print("\n[3/5] Verifying Training Hyperparameters in Code...")
    expected_params = [
        ("imgsz", "960"),
        ("epochs", "100"),
        ("patience", "25"),
        ("mosaic", "1.0"),
        ("mixup", "0.15"),
        ("copy_paste", "0.0"),
        ("close_mosaic", "15"),
        ("optimizer", "AdamW"),
        ("seed", "42"),
        ("save_period", "5"),
    ]
    for param, val in expected_params:
        assert f"'{param}': {val}" in all_code or f"'{param}': '{val}'" in all_code, f"Hyperparameter {param}={val} missing from code!"
        print(f"      Verified: {param} = {val}")

    # 4. Check Batch Fallback
    print("\n[4/5] Verifying Batch Fallback...")
    assert "initial_batch = 16" in all_code, "Initial batch 16 missing!"
    assert "batch=8" in all_code, "Fallback to batch 8 missing!"
    assert "batch=4" in all_code, "Fallback to batch 4 missing!"
    assert "accumulate" not in all_code, "accumulate parameter should NOT be present!"
    print("      Verified: Safe Batch Fallback Cascade (16 -> 8 -> 4).")

    # 5. Check Output and Checkpoint Export Paths
    print("\n[5/5] Verifying Artifact Export Logic...")
    assert "shutil.copy(best_weight, working_dir / 'yolo11m_best.pt')" in all_code, "Checkpoint export missing!"
    assert "metrics_summary.json" in all_code, "Metrics summary JSON missing!"
    assert "/kaggle/working" in all_code, "/kaggle/working directory path missing!"
    print("      Verified: Artifacts are copied to /kaggle/working/ for automated download.")

    print("\n" + "=" * 70)
    print("  PHASE 2 AUDIT PASSED — ALL 5 AUDIT GATES VERIFIED!")
    print("=" * 70)

    # Save audit report
    audit_report = {
        "phase": "PHASE 2 — TRAINING NOTEBOOK AUDIT",
        "kernel_id": meta.get('id'),
        "notebook_file": str(nb_path),
        "total_cells": len(nb['cells']),
        "gpu_enabled": True,
        "internet_enabled": True,
        "is_private": True,
        "dataset_attached": "mishajaiswal0511/ecovision-train-val",
        "hyperparameters": {p: v for p, v in expected_params},
        "batch_fallback": "16 -> 8 -> 4",
        "checkpoint_export": "yolo11m_best.pt to /kaggle/working/",
        "status": "PASSED"
    }
    with open("docs/phase2_notebook_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

if __name__ == "__main__":
    audit_notebook()

