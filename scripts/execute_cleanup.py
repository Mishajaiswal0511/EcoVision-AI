import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path("D:/Projects/ml_project/472")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
QUARANTINE_DIR = ROOT / "backups" / f"pre_cleanup_quarantine_{TIMESTAMP}"
ARCHIVE_EPOCHS_DIR = ROOT / "archive" / "models" / "yolo11m_v1_epochs"
STRAY_IMAGES_DIR = ROOT / "data" / "real_world_eval" / "sample_test_images"
HISTORICAL_REPORTS_DIR = ROOT / "docs" / "historical_reports"

def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print(f"=== Starting Safe Cleanup Execution at {datetime.now().isoformat()} ===")
    
    # 0. Safety verification of production models first
    v5_best = ROOT / "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt"
    baseline_best = ROOT / "models/weights_best.pt"
    
    v5_hash_pre = sha256_file(v5_best)
    baseline_hash_pre = sha256_file(baseline_best)
    
    print(f"Verified Active V5 Model Hash: {v5_hash_pre}")
    print(f"Verified Baseline 6-Class Model Hash: {baseline_hash_pre}")
    assert v5_hash_pre == "bf3890b8374bfae6c376a898fc2080f79095aae0e9bd37ed612e86fb1dffc66a", f"Unexpected V5 hash: {v5_hash_pre}"
    assert baseline_hash_pre == "1933635e9c9ca2c26c68204cc8ceae7950609386ddcc394a6731185bef694d87", f"Unexpected Baseline hash: {baseline_hash_pre}"
    
    # Create directories
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_EPOCHS_DIR.mkdir(parents=True, exist_ok=True)
    STRAY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    moves = []
    
    # 1. 15 Intermediate Epoch checkpoints
    epochs_src_dir = ROOT / "models/experiments/yolo11m_v1/runs/train/yolo11m_waste_960/weights"
    for epoch_file in epochs_src_dir.glob("epoch*.pt"):
        moves.append({
            "category": "epoch_archive",
            "source": epoch_file,
            "destination": ARCHIVE_EPOCHS_DIR / epoch_file.name
        })
        
    # 2. Stray test images
    stray_1 = ROOT / "models/compost-leftover-food.webp"
    stray_2 = ROOT / "models/large-pile-of-overflowing.jpg"
    if stray_1.exists():
        moves.append({
            "category": "stray_images",
            "source": stray_1,
            "destination": STRAY_IMAGES_DIR / stray_1.name
        })
    if stray_2.exists():
        moves.append({
            "category": "stray_images",
            "source": stray_2,
            "destination": STRAY_IMAGES_DIR / stray_2.name
        })
        
    # 3. Duplicate ZIPs and unreferenced artifacts -> Quarantine
    quarantine_files = [
        ROOT / "ecovision_bbox_v2.zip",
        ROOT / "data/ecovision_train_val.zip",
        ROOT / "yolo26n.pt",
        ROOT / "yolo11s_waste_960.pt.part"
    ]
    for qf in quarantine_files:
        if qf.exists():
            moves.append({
                "category": "quarantine",
                "source": qf,
                "destination": QUARANTINE_DIR / qf.name
            })
            
    # 4. 13 Root historical reports -> docs/historical_reports/
    reports = [
        "COMPLETION_REPORT.md",
        "DATASET_AUDIT_REPORT.md",
        "FINAL_COMPLETION_REPORT.md",
        "FINAL_DEMO_CHECKLIST.md",
        "FINAL_FRONTEND_INTEGRATION_REPORT.md",
        "FINAL_STATUS_REPORT.md",
        "MODEL_CARD.md",
        "MODEL_COMPARISON_REPORT.md",
        "NEXT_STEPS.md",
        "PHASE1_COMPLETION_REPORT.md",
        "PROJECT_PLAN.md",
        "REAL_WORLD_FAILURE_ANALYSIS.md",
        "SESSION_COMPLETION_REPORT.md"
    ]
    for rep in reports:
        rep_path = ROOT / rep
        if rep_path.exists():
            moves.append({
                "category": "historical_reports",
                "source": rep_path,
                "destination": HISTORICAL_REPORTS_DIR / rep
            })
            
    print(f"\nPrepared {len(moves)} items for relocation/quarantine.")
    
    # 5. Pre-compute SHA-256 for all files to be moved
    manifest_records = []
    for item in moves:
        src = item["source"]
        dst = item["destination"]
        src_sha = sha256_file(src)
        manifest_records.append({
            "category": item["category"],
            "source": str(src.relative_to(ROOT)),
            "destination": str(dst.relative_to(ROOT)),
            "sha256": src_sha,
            "size_bytes": src.stat().st_size
        })
        
    # Execute moves and verify hashes
    print("\nExecuting moves and verifying SHA-256 hashes...")
    verified_moves = []
    for record in manifest_records:
        src = ROOT / record["source"]
        dst = ROOT / record["destination"]
        expected_sha = record["sha256"]
        
        # Move file
        shutil.move(str(src), str(dst))
        
        # Verify SHA-256 at destination
        dst_sha = sha256_file(dst)
        if dst_sha != expected_sha:
            raise RuntimeError(f"FATAL: Hash mismatch for {src} -> {dst}! Expected: {expected_sha}, got: {dst_sha}")
        
        verified_moves.append(record)
        print(f" [OK] {record['source']} -> {record['destination']} ({record['size_bytes']} bytes, SHA match)")
        
    # 6. Save rollback manifest JSON
    rollback_manifest = {
        "timestamp": TIMESTAMP,
        "quarantine_dir": str(QUARANTINE_DIR.relative_to(ROOT)),
        "moves": verified_moves,
        "protected_models": {
            "v5_best": {
                "path": "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt",
                "sha256": v5_hash_pre
            },
            "baseline_best": {
                "path": "models/weights_best.pt",
                "sha256": baseline_hash_pre
            }
        }
    }
    manifest_path = ROOT / "backups" / f"rollback_manifest_{TIMESTAMP}.json"
    manifest_path.write_text(json.dumps(rollback_manifest, indent=2), encoding="utf-8")
    print(f"\nWrote verified rollback manifest to {manifest_path}")
    
    # 7. Generate PowerShell rollback script
    ps1_lines = [
        f"# Automated Rollback Script for Cleanup {TIMESTAMP}",
        "# Run this script in PowerShell to completely reverse all moves.",
        "$ErrorActionPreference = 'Stop'",
        f"Write-Host 'Starting Rollback for cleanup {TIMESTAMP}...'",
        ""
    ]
    for record in reversed(verified_moves):
        src_rel = record["source"].replace("/", "\\")
        dst_rel = record["destination"].replace("/", "\\")
        ps1_lines.append(f"if (Test-Path '{dst_rel}') {{")
        ps1_lines.append(f"    $parent = Split-Path '{src_rel}' -Parent")
        ps1_lines.append(f"    if ($parent -and -not (Test-Path $parent)) {{ New-Item -ItemType Directory -Path $parent -Force | Out-Null }}")
        ps1_lines.append(f"    Move-Item -Path '{dst_rel}' -Destination '{src_rel}' -Force")
        ps1_lines.append(f"    Write-Host 'Restored: {src_rel}'")
        ps1_lines.append(f"}}")
    ps1_lines.append("Write-Host 'Rollback complete. All items restored to exact original locations.'")
    
    rollback_script_path = ROOT / "scripts" / "rollback_cleanup.ps1"
    rollback_script_path.write_text("\n".join(ps1_lines), encoding="utf-8")
    print(f"Wrote rollback script to {rollback_script_path}")
    
    # 8. Clean __pycache__ and .pytest_cache (regenerable)
    caches_removed = 0
    for p in ROOT.rglob("__pycache__"):
        if ".venv" not in p.parts:
            shutil.rmtree(p, ignore_errors=True)
            caches_removed += 1
    pytest_cache = ROOT / ".pytest_cache"
    if pytest_cache.exists():
        shutil.rmtree(pytest_cache, ignore_errors=True)
        caches_removed += 1
    print(f"Cleaned {caches_removed} cache directories (__pycache__ / .pytest_cache).")
    
    # 9. Update .gitignore safely
    gitignore_path = ROOT / ".gitignore"
    if gitignore_path.exists():
        gi_content = gitignore_path.read_text(encoding="utf-8")
    else:
        gi_content = ""
        
    additions = []
    for pattern in ["archive/", "backups/", "scratch/"]:
        if pattern not in gi_content:
            additions.append(pattern)
            
    if additions:
        gi_content += "\n# Cleanup archives and backups\n" + "\n".join(additions) + "\n"
        gitignore_path.write_text(gi_content, encoding="utf-8")
        print("Updated .gitignore with archive/ and backups/ patterns.")
        
    # 10. Post-cleanup verification of production models
    v5_hash_post = sha256_file(v5_best)
    baseline_hash_post = sha256_file(baseline_best)
    assert v5_hash_pre == v5_hash_post, "FATAL: V5 model was altered during cleanup!"
    assert baseline_hash_pre == baseline_hash_post, "FATAL: Baseline model was altered during cleanup!"
    print("Post-cleanup check: Both production model weights are 100% untouched and verified identical.")
    print("=== Safe Cleanup Execution Complete ===")

if __name__ == "__main__":
    main()
