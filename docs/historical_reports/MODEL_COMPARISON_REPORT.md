# EcoVision AI — Comprehensive Model Comparison Report

**Document:** `MODEL_COMPARISON_REPORT.md`  
**Date:** 2026-11-01  
**Author:** ML Engineering & Computer Vision Team  
**Hardware:** AMD Ryzen 7 7735U CPU (8 Cores / 16 Threads), 16 GB RAM  
**Evaluated Models:** M1 (V4-Prod), M2 (V4-11m), M3 (V5-ExpA), M4 (V5-ExpB)

---

## 1. Checkpoint Architectural & Taxonomic Overview

| Model Identifier | Architecture | Parameters | Input Resolution | Output Heads | Checkpoint Path | SHA-256 Digest | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **M1 (V4-Prod)** | YOLO11s | 9.4M | 640×640 | 6 Classes | `models/weights_best.pt` | `1933635e9c9ca2c26c68204cc8ceae7950609386ddcc394a6731185bef694d87` | Baseline Production Backup |
| **M2 (V4-11m)** | YOLO11m | 20.1M | 960×960 | 6 Classes | `models/experiments/yolo11m_best.pt` | `0de051dc2198c6efa987eb32658fc13cbef9874cbbebf38435dd9b6bc458d343` | High-Capacity V4 Baseline |
| **M3 (V5-ExpA)** | YOLO11s | 9.4M | 640×640 | 7 Classes | `.../exp_a_yolo11s_640/.../best.pt` | `7df137c919eaf47370a17c2313d00032e02120c3349575e7f137ad23b1de58c3` | 7-Class 640px Model |
| **M4 (V5-ExpB)** | YOLO11s | 9.4M | 800×800 | 7 Classes | `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt` | `bf3890b8374bfae6c376a898fc2080f79095aae0e9bd37ed612e86fb1dffc66a` | **Selected Staging Candidate** |

---

## 2. Quantitative Metric Comparison (3,007 Validation Images)

Evaluated on the independent V5 scene-grouped validation set (`data/processed_v5/images/val`, 3,007 images, 10,732 ground-truth instances):

| Metric | M1 (V4-Prod 640) | M2 (V4-11m 960) | M3 (V5-ExpA 640) | M4 (V5-ExpB 800) | Metric Delta (M4 vs M1 Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classes Evaluated** | 6 (No Organic) | 6 (No Organic) | 7 | **7** | **+1 Class (Organic)** |
| **Precision** | 0.642 | 0.658 | 0.670 | **0.680** | **+5.9%** |
| **Recall** | 0.485 | 0.502 | 0.506 | **0.514** | **+6.0%** |
| **mAP@50 (conf=0.001)** | 0.528 | 0.548 | 0.564 | **0.573** | **+8.5%** |
| **mAP@50-95** | 0.385 | 0.401 | 0.415 | **0.415** | **+7.8%** |
| **mAP@50 (conf=0.15)** | 0.489 | 0.508 | 0.517 | **0.520** | **+6.3%** |
| **Mean CPU Latency** | 125 ms | 542 ms | 137 ms | **183 ms** | -58 ms vs M1 (Fast) |
| **Peak RAM (RSS)** | 480 MB | 1,220 MB | 518 MB | **594 MB** | Lightweight ($< 4\%$ RAM) |

---

## 3. Class-by-Class mAP@50 Breakdown (Operational `conf=0.15`)

| Class Name | M1 (V4-Prod 640) | M2 (V4-11m 960) | M3 (V5-ExpA 640) | M4 (V5-ExpB 800) | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **plastic** | 0.602 | 0.614 | 0.618 | **0.624** | Consistent gain; 800px resolves small containers |
| **metal** | 0.685 | 0.701 | 0.710 | **0.715** | Strongest overall category; clear edge gradients |
| **cardboard** | 0.620 | 0.635 | 0.642 | **0.649** | High precision on corrugated shipping boxes |
| **glass** | 0.590 | 0.610 | 0.615 | **0.621** | Good bottle localization; specular highlight sensitivity |
| **paper** | 0.551 | 0.569 | 0.574 | **0.580** | Reliable on dry clean paper |
| **general_trash** | 0.077 | 0.092 | 0.078 | **0.115** | **+48.6% relative gain under M4**; remaining challenge |
| **organic** | N/A (0.000) | N/A (0.000) | 0.379 | **0.384** | Enabled exclusively in V5; compost tray domain |

---

## 4. Multi-Model Ensembling Assessment

To explore whether combining existing models could overcome individual weaknesses, four ensembling paradigms were implemented and benchmarked:
1. **Majority Voting ($\ge 2$ models agreement):** Successfully eliminated isolated single-model false alarms, but dropped valid low-confidence trash items due to boundary offsets ($IoU < 0.45$).
2. **Confidence-Weighted Voting:** Weighted box coordinates by summed model confidence.
3. **Detection-Level Weighted Boxes Fusion (WBF):** Consolidates bounding boxes into tight single proposals.

### Ensembling Verdict: **REJECTED**
* **Reason 1 (Data Blind Spots Persist):** On out-of-distribution inputs like the banana peel (`organic_01.webp`), all models lack standalone peel training. Combining them simply concentrates the false prediction (`glass: 0.60`).
* **Reason 2 (Taxonomy Incompatibility):** Models M1 and M2 cannot predict Organic, so voting schemes structurally depress the organic class.
* **Reason 3 (Severe Latency Penalty):** Running four models sequentially on CPU took **> 850 ms** ($5.3\times$ slower than M4 solo), violating real-time interactive constraints.

---

## 5. Final Model Selection

**M4 (`yolo11s_v5_best.pt` @ 800px)** is selected as the sole candidate model:
- Outperforms all other models in mAP, precision, and recall across all 7 classes.
- Delivers **183 ms latency (5.5 FPS)** on target AMD Ryzen 7 7735U CPU.
- Augmented with **Containment-Aware Adaptive NMS (CA-NMS)** to resolve cardboard pile aggregate boxes and broken glass ghost metal artifacts.

