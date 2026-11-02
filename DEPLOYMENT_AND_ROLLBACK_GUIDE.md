# EcoVision AI — Production Promotion & Rollback Guide

**Document:** `DEPLOYMENT_AND_ROLLBACK_GUIDE.md`  
**Current Status:** Staging Candidate Validated (Production Assets Frozen Awaiting Formal Authorization)  
**Candidate Model:** M4 (`models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`, SHA-256: `bf3890b8...`)  
**Production Baseline:** M1 (`models/weights_best.pt`, SHA-256: `1933635e...`)

---

## 1. Safety & Authorization Gate

> [!IMPORTANT]
> In strict compliance with the Master Safety Rules, **zero production files or checkpoints have been replaced**. The candidate model is verified and live in the isolated staging environment. Production promotion must only be initiated when the user explicitly issues deployment authorization.

---

## 2. Production Promotion Procedure (When Authorized)

When formal authorization is granted, execute the following steps in sequence:

### Step 1: Secure Production Checkpoint Backup
```powershell
# Create an immutable backup of the original V4 production weights
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item -Path "models\weights_best.pt" -Destination "models\weights_best_v4_backup_$timestamp.pt"

# Verify backup SHA-256 matches original: 1933635e9c9ca2c26c68204cc8ceae7950609386ddcc394a6731185bef694d87
Get-FileHash -Algorithm SHA256 "models\weights_best_v4_backup_$timestamp.pt"
```

### Step 2: Promote M4 Checkpoint to Production Path
```powershell
# Copy the verified V5 candidate model to the production checkpoint location
Copy-Item -Force -Path "models\experiments\v5_taxonomy\training\yolo11s_v5_best.pt" -Destination "models\weights_best.pt"

# Verify new production SHA-256 matches M4: bf3890b8374bfae6c376a898fc2080f79095aae0e9bd37ed612e86fb1dffc66a
Get-FileHash -Algorithm SHA256 "models\weights_best.pt"
```

### Step 3: Promote Staging Services to Production
```powershell
# Promote the audited disposal guide
Copy-Item -Force -Path "models\experiments\v5_taxonomy\preproduction\staging\backend\data\disposal_guide.json" -Destination "docs\disposal_guide.json"

# Promote CA-NMS and collision-free inference service to production backend
Copy-Item -Force -Path "models\experiments\v5_taxonomy\preproduction\staging\backend\services\inference.py" -Destination "backend\services\inference.py"
```

### Step 4: Restart Production Services
```powershell
# Restart FastAPI backend on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Re-run automated regression test suite
python scripts\test_staging_demo_workflow.py
```

---

## 3. Instant Rollback Procedure

If any regression, latency anomaly, or operational defect is detected in production, execute this one-command rollback sequence to instantly return to the original V4 production state:

```powershell
# 1. Restore original production checkpoint
Copy-Item -Force -Path "models\weights_best_v4_backup_*.pt" -Destination "models\weights_best.pt"

# 2. Restore V4 legacy inference service
Copy-Item -Force -Path "backend\services\inference_v4_backup.py" -Destination "backend\services\inference.py"

# 3. Restore V4 6-class frontend constants
Copy-Item -Force -Path "frontend\src\constants\classes_v4_backup.js" -Destination "frontend\src\constants\classes.js"

# 4. Verify restored production hash
$hash = (Get-FileHash -Algorithm SHA256 "models\weights_best.pt").Hash
if ($hash -eq "1933635E9C9CA2C26C68204CC8CEAE7950609386DDCC394A6731185BEF694D87") {
    Write-Host "ROLLBACK SUCCESSFUL: Production model restored to original baseline." -ForegroundColor Green
} else {
    Write-Host "ROLLBACK VERIFICATION FAILED: Checksum mismatch!" -ForegroundColor Red
}

# 5. Restart server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## 4. Disaster Recovery & Sealed Data Integrity

* **Sealed Real-World Evaluation:** `data/real_world_eval/` remains sealed throughout all deployment and rollback procedures. It is never accessed, mounted, or read by deployment scripts.
* **Service Downtime:** Transition time between staging and production is $< 5$ seconds during uvicorn reload.

