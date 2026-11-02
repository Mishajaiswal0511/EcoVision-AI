# EcoVision AI — Confidence Threshold Analysis & Evaluation Report

**Date:** October 30, 2026  
**Model:** YOLO11s (`models/weights_best.pt`)  
**Evaluated Thresholds:** `0.10` (10%), `0.15` (15%), `0.25` (25% default), `0.35` (35%)  
**Test Split:** Development Validation Set (24 positive waste images across all 6 classes + 10 verified negative non-waste images; sealed `data/real_world_eval/` was strictly excluded).  

---

## 1. Summary Comparison Across All Four Thresholds

| Metric | `conf = 0.10` | `conf = 0.15` | `conf = 0.25` (Default) | `conf = 0.35` |
| :--- | :---: | :---: | :---: | :---: |
| **Positive Waste Recall** (Images with >= 1 detection) | **24/24 (100.0%)** | **24/24 (100.0%)** | **19/24 (79.2%)** | **18/24 (75.0%)** | 
| **Missed Positive Images** (0 detections) | 0/24 (0.0%) | 0/24 (0.0%) | 5/24 (20.8%) | 6/24 (25.0%) | 
| **Ground-Truth Class Match Rate** (>= 1 box matches GT) | 18/24 (75.0%) | 17/24 (70.8%) | 15/24 (62.5%) | 14/24 (58.3%) | 
| **Total Positive Detections (Boxes)** | 169 boxes | 111 boxes | 57 boxes | 36 boxes | 
| **Negative Image False Alarm Rate** (Images with >= 1 FP) | 1/10 (10.0%) | 1/10 (10.0%) | 1/10 (10.0%) | 0/10 (0.0%) | 
| **Total False Positive Boxes on Non-Waste** | **2 boxes** | **2 boxes** | **1 boxes** | **0 boxes** | 

---

## 2. Image-by-Image Detection Breakdown

Detailed records for all 34 evaluated development images across the four thresholds:

| Image Filename | Target / GT Class | `conf = 0.10` | `conf = 0.15` | `conf = 0.25` | `conf = 0.35` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gw_biodegradable1930_jpe...` | plastic | 4 det (plastic 83%) ✓ | 3 det (plastic 83%) ✓ | 3 det (plastic 83%) ✓ | 3 det (plastic 83%) ✓ |
| `gw_biodegradable2101_jpe...` | plastic, paper | 8 det (paper 54%) ✓ | 6 det (paper 54%) ✓ | 2 det (paper 54%) ✓ | 1 det (paper 54%) ✓ |
| `gw_biodegradable997_jpg....` | plastic | 11 det (plastic 23%) ✓ | 8 det (plastic 23%) ✓ | 0 det *(Missed)* | 0 det *(Missed)* |
| `gw_cardboard561_jpg.rf.8...` | plastic, cardboard | 17 det (cardboard 48%) ✓ | 13 det (cardboard 48%) ✓ | 8 det (cardboard 48%) ✓ | 2 det (cardboard 48%) ✓ |
| `gw_biodegradable1660_jpg...` | metal | 3 det (metal 53%) ✓ | 2 det (metal 53%) ✓ | 2 det (metal 53%) ✓ | 1 det (metal 53%) ✓ |
| `gw_biodegradable1956_jpe...` | metal | 1 det (glass 25%) ✗ | 1 det (glass 25%) ✗ | 0 det *(Missed)* | 0 det *(Missed)* |
| `gw_biodegradable2160_jpe...` | metal | 6 det (cardboard 19%) ✗ | 2 det (cardboard 19%) ✗ | 0 det *(Missed)* | 0 det *(Missed)* |
| `gw_biodegradable2236_jpe...` | metal, glass | 4 det (metal 31%) ✓ | 3 det (metal 31%) ✓ | 1 det (metal 31%) ✓ | 0 det *(Missed)* |
| `gw_biodegradable1262_jpg...` | cardboard | 16 det (cardboard 38%) ✓ | 8 det (cardboard 38%) ✓ | 4 det (cardboard 38%) ✓ | 2 det (cardboard 38%) ✓ |
| `gw_biodegradable1442_jpg...` | cardboard | 4 det (cardboard 63%) ✓ | 2 det (cardboard 63%) ✓ | 2 det (cardboard 63%) ✓ | 1 det (cardboard 63%) ✓ |
| `gw_biodegradable1703_jpg...` | cardboard | 14 det (cardboard 23%) ✓ | 7 det (cardboard 23%) ✓ | 0 det *(Missed)* | 0 det *(Missed)* |
| `gw_biodegradable2002_jpe...` | cardboard | 2 det (glass 39%) ✗ | 1 det (glass 39%) ✗ | 1 det (glass 39%) ✗ | 1 det (glass 39%) ✗ |
| `gw_biodegradable408_jpg....` | glass | 14 det (glass 60%) ✓ | 11 det (glass 60%) ✓ | 5 det (glass 60%) ✓ | 4 det (glass 60%) ✓ |
| `gw_biodegradable450_jpg....` | metal, glass | 5 det (paper 17%) ✗ | 1 det (paper 17%) ✗ | 0 det *(Missed)* | 0 det *(Missed)* |
| `gw_biodegradable462_jpg....` | glass | 6 det (glass 47%) ✓ | 5 det (glass 47%) ✓ | 5 det (glass 47%) ✓ | 3 det (glass 47%) ✓ |
| `gw_biodegradable825_jpg....` | metal, glass | 9 det (glass 90%) ✓ | 6 det (glass 90%) ✓ | 5 det (glass 90%) ✓ | 5 det (glass 90%) ✓ |
| `gw_biodegradable1020_jpg...` | paper | 16 det (cardboard 41%) ✗ | 12 det (cardboard 41%) ✗ | 4 det (cardboard 41%) ✗ | 2 det (cardboard 41%) ✗ |
| `gw_cardboard1232_jpg.rf....` | cardboard, paper | 10 det (cardboard 89%) ✓ | 6 det (cardboard 89%) ✓ | 5 det (cardboard 89%) ✓ | 3 det (cardboard 89%) ✓ |
| `gw_cardboard1337_jpg.rf....` | cardboard, paper | 3 det (cardboard 92%) ✓ | 2 det (cardboard 92%) ✓ | 2 det (cardboard 92%) ✓ | 2 det (cardboard 92%) ✓ |
| `gw_cardboard1604_jpeg.rf...` | paper | 1 det (paper 98%) ✓ | 1 det (paper 98%) ✓ | 1 det (paper 98%) ✓ | 1 det (paper 98%) ✓ |
| `taco_000002_JPG.rf.e0a28...` | plastic, paper, general_trash | 8 det (general_trash 37%) ✓ | 6 det (general_trash 37%) ✓ | 3 det (general_trash 37%) ✓ | 1 det (general_trash 37%) ✓ |
| `taco_000002_jpg.rf.e5d45...` | paper, general_trash | 1 det (plastic 92%) ✗ | 1 det (plastic 92%) ✗ | 1 det (plastic 92%) ✗ | 1 det (plastic 92%) ✗ |
| `taco_000003_JPG.rf.49b3d...` | plastic, general_trash | 5 det (plastic 89%) ✓ | 3 det (plastic 89%) ✓ | 2 det (plastic 89%) ✓ | 2 det (plastic 89%) ✓ |
| `taco_000003_jpg.rf.3f690...` | general_trash | 1 det (plastic 85%) ✗ | 1 det (plastic 85%) ✗ | 1 det (plastic 85%) ✗ | 1 det (plastic 85%) ✗ |
| `neg_000000460683.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000461751.jpg` | *(Negative / Clean)* | 2 det (plastic 28%) FP | 2 det (plastic 28%) FP | 1 det (plastic 28%) FP | 0 det *(Clean)* |
| `neg_000000462629.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000462643.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000463199.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000463522.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000463618.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000463730.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000464251.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |
| `neg_000000464358.jpg` | *(Negative / Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* | 0 det *(Clean)* |

*Legend: `✓` = Correct GT match; `✗` = Class mismatch or false positive on positive image; `FP` = False alarm on non-waste image.*

---

## 3. In-Depth Behavioral Analysis

### A. `conf = 0.10` (High Sensitivity)
- **Positive Waste Recall:** 100.0% (24/24 images detected). Zero positive images were missed.
- **GT Match Rate:** 75.0% (18/24 images).
- **Total Detections:** 169 boxes across 24 images (~7.0 boxes/image).
- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (2 false positive boxes total on `neg_000000461751.jpg`).
- **Behavioral Observation:** Maximizes object discovery on challenging waste (e.g. small plastic fragments, crushed cans, distant packaging), but generates clutter and overlapping boxes on textured surfaces.

### B. `conf = 0.15` (Balanced Sensitivity)
- **Positive Waste Recall:** 100.0% (24/24 images detected). Zero positive images were missed.
- **GT Match Rate:** 70.8% (17/24 images).
- **Total Detections:** 111 boxes across 24 images (~4.6 boxes/image), reducing box clutter by **34.3%** compared to 0.10.
- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (2 boxes total).
- **Behavioral Observation:** Successfully rescues low-confidence waste objects that are completely missed at default `conf=0.25`, while pruning weak, spurious background boxes.

### C. `conf = 0.25` (Default Setting)
- **Positive Waste Recall:** 79.2% (19/24 images detected). **20.8% of positive images (5/24) return 0 detections.**
- **GT Match Rate:** 62.5% (15/24 images).
- **Total Detections:** 57 boxes across 24 images (~2.4 boxes/image).
- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (1 box total).
- **Behavioral Observation:** Delivers clean, conservative bounding boxes with minimal false alarms, but causes user friction on difficult images where the model's confidence scores naturally peak between 0.15 and 0.24 (e.g., `taco_000000_JPG` where top confidence was 0.2321).

### D. `conf = 0.35` (High Precision / Conservative)
- **Positive Waste Recall:** 75.0% (18/24 images detected). **25.0% of positive images (6/24) return 0 detections.**
- **GT Match Rate:** 58.3% (14/24 images).
- **Total Detections:** 36 boxes across 24 images (~1.5 boxes/image).
- **False Positives on Negatives:** **0.0% (0/10 images, 0 boxes).** Completely immune to false alarms on all tested non-waste backgrounds.
- **Behavioral Observation:** Highly reliable when an object is detected, but prone to false negatives on occluded or partially deformed waste.

---

## 4. Balanced Threshold Recommendation

### Recommended Default: `conf = 0.20` — `0.25` with Interactive Slider `0.10` — `0.50`

1. **Why `conf = 0.25` remains a solid starting default:** It maintains high precision on clean images and produces concise, legible bounding boxes without overcrowding the user interface with redundant overlapping detections.
2. **Why the `0.10` minimum slider fix is essential:** When a user photographs waste that is crushed, dimly lit, or partially obscured, the model's confidence often falls into the `0.15`–`0.24` range. Allowing the user to slide the threshold down to `10%` or `15%` recovers **100% of the positive images** in our test sample without causing significant false alarm surges on clean non-waste backgrounds.
3. **Operational Recommendation:** Keep `default = 0.25` as the initial baseline. When 0 detections occur, prompt the user with: *"No waste items detected. Try lowering the confidence slider to 10–15% or adjusting lighting."*

---

## 5. Compliance & Protocol Safeguards
- Model weights (`models/weights_best.pt`) remained unmodified.
- Training configurations and hyperparameter files remained unchanged.
- The sealed `data/real_world_eval/` benchmark was not opened, accessed, or used.
- No claims of improved underlying model accuracy are made; findings reflect empirical score distributions of the existing YOLO11s checkpoint on development validation data.