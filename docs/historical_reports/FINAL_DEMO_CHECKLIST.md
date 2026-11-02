# EcoVision AI — Final Academic Demonstration Checklist

**Target System:** EcoVision AI Staging Candidate (YOLO11s V5 @ 800×800)  
**Selected Checkpoint:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt` (M4, SHA-256: `bf3890b8...`)  
**Host Architecture:** AMD Ryzen 7 7735U CPU Inference, 16 GB RAM  
**Evaluation Status:** 10/10 Workflow Tests Verified (100% Pass Rate)

---

## 1. Candidate Configuration Baseline

* **Model Architecture:** Ultralytics YOLO11s (9.4M parameters)
* **Active Checkpoint Path:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`
* **Inference Resolution:** `800×800` (imgsz=800)
* **Default Confidence Threshold:** `0.15` (15%)
* **Default NMS IoU Threshold:** `0.45` (45%)
* **Class-Agnostic NMS:** `True` (enabled by default)
* **Taxonomy (7 Classes):**
  - `0`: plastic (Blue Bin / Dry Recyclables)
  - `1`: metal (Blue Bin / Scrap Collection)
  - `2`: cardboard (Blue Bin / Dry Recyclables)
  - `3`: glass (Blue Bin / Dedicated Glass Bank with safety wrap)
  - `4`: paper (Blue Bin / Dry Recyclables)
  - `5`: general_trash (Black/Grey Bin / Residual Waste with MoEFCC Sanitary Advisory)
  - `6`: organic (Green Bin / Wet Waste / Compost)

---

## 2. Verified Pre-Demo Test Results

The automated staging verification suite (`scripts/test_staging_demo_workflow.py`) verified 10 core integration tests against the live staging server:

| Test # | Test Case Description | Verified Behavior | Status |
| :---: | :--- | :--- | :---: |
| **1** | Health & Model Readiness API | Returns `200 OK`, `status="healthy"`, 7 classes verified, M4 loaded | **PASS** |
| **2** | Single Plastic Object Detection | `gw_plastic1002` detected with 94.0% top confidence, Blue Bin guidance | **PASS** |
| **3** | Glass Object & Safety Guidance | `gw_glass1000` detected; includes broken glass safety wrapping notice | **PASS** |
| **4** | Organic Waste Detection | `gw_biodegradable1225` detected; directs to Green Bin (compost) | **PASS** |
| **5** | General Trash & Sanitary Advisory | `taco_000000` detected; displays statutory MoEFCC red-marked pouch warning | **PASS** |
| **6** | Multi-Object Waste Scene | `gw_biodegradable1194` localized 14 distinct items with bounding boxes | **PASS** |
| **7** | Corrupted Image Handling | Injected corrupt bytes; returns clean `HTTP 400` with descriptive error | **PASS** |
| **8** | Unsupported File Format | Plain text document uploaded; rejected with `HTTP 400` without crash | **PASS** |
| **9** | Empty (0-byte) Buffer Handling | 0-byte file uploaded; rejected with `HTTP 400` cleanly | **PASS** |
| **10** | Direct Disposal Category API | All 7 classes verified via `/api/disposal/{class}`; invalid query returns 404 | **PASS** |

---

## 3. How to Launch and Run the Staging Demo

### Step 1: Start the Staging Backend
Open a terminal in the project root (`D:\Projects\ml_project\472`) with the Python virtual environment activated:
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start isolated staging FastAPI server on port 8000
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir models/experiments/v5_taxonomy/preproduction/staging/backend
```
*Verify output:*
```text
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Start the Frontend UI
Open a second terminal in the project root:
```powershell
# Launch Vite frontend development server
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```
*Verify output:*
```text
  VITE v8.3.5  ready in 1670 ms
  ➜  Local:   http://127.0.0.1:5173/
```

### Step 3: Open the Web Application
Open your web browser and navigate to:
**`http://127.0.0.1:5173/`**

---

## 4. Live Demonstration Walkthrough Script

### Act 1: System Health & Architecture Overview
- [ ] Point out the header badge indicating **Model: V5 (7-Class @ 800px)**.
- [ ] Show the 7 supported classes in the UI: Plastic, Metal, Cardboard, Glass, Paper, General Trash, Organic.
- [ ] Confirm backend connectivity indicator is active.

### Act 2: Standard Recyclable & Organic Detection
- [ ] **Plastic / Can Detection:** Upload a test recyclable image (e.g. from `data/processed_v5/images/val/gw_plastic1002_jpg...`).
  - Demonstrate color-coded bounding boxes.
  - Review confidence score pill.
  - Review Blue Bin segregation guidance.
- [ ] **Organic Waste Detection:** Upload `data/processed_v5/images/val/gw_biodegradable1225_jpg...`.
  - Highlight Class 6 (`organic`) directing to Green Bin (Wet Waste).
  - Emphasize that Organic is an upgrade unique to V5 (not present in V4).

### Act 3: Statutory Disposal & Safety Guidelines
- [ ] **Broken Glass Handling:** Upload `data/processed_v5/images/val/gw_glass1000_jpg...`.
  - Show the amber safety alert: requirement to wrap shattered glass in newspaper before municipal handover.
- [ ] **Sanitary Waste Guidance:** Upload `data/processed_v5/images/val/taco_000000_jpg...`.
  - Show the MoEFCC 2026 red-marked newspaper wrap protocol for personal hygiene items.
- [ ] **Conditional Packaging:** Explain takeaway container and greasy pizza box conditional rules.

### Act 4: Multi-Object Detection & Threshold Controls
- [ ] **Dense Scene:** Upload `data/processed_v5/images/val/gw_biodegradable1194_jpg...`.
  - Highlight 14 localized instances.
  - Switch between "Annotated Prediction" and "Original Upload" tabs.
- [ ] **Confidence Slider:** Adjust threshold from 0.15 to 0.40; demonstrate filtering of lower-confidence proposals.
- [ ] **Class-Agnostic NMS Toggle:** Demonstrate suppression of overlapping proposals across distinct classes.

### Act 5: Robust Error Handling Demonstration
- [ ] **Corrupt / Invalid Upload:** Attempt to upload an invalid file (e.g., text document or corrupted byte stream).
  - Demonstrate that the UI displays a clean rose-colored error banner (`"Unsupported file format..."` or `"Invalid or unreadable image..."`).
  - Emphasize that the server remains responsive and does not crash or throw uncaught 500 exceptions.

### Act 6: Academic Transparency on Limitations
- [ ] Disclose the documented limitations in [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md):
  - Banana peel visual confusion with Glass/Plastic due to specular reflections and training distribution limits.
  - General trash intra-class variance and low recall.
  - Explain why multi-model ensembling failed to fix these and why dataset expansion (Tier 2 / V5.1) is the proper engineering solution.

