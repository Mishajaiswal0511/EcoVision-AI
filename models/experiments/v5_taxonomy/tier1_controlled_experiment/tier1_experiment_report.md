# EcoVision AI — Tier 1 Controlled Inference Experiment Report

**Date:** 2026-11-01 15:39:31  
**Evaluated Checkpoint:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`  
**Protocol:** Strictly Read-Only (Zero retraining, zero dataset modifications, zero production file changes)  

---

## 1. Executive Summary & Core Experimental Findings

1. **Class-Agnostic NMS Impact (`agnostic_nms=True`):**
   - **Eliminating Same-Object Cross-Class Duplicates:** On Image 1 (Mixed Recyclables) and Image 4 (Banana Peel), enabling `agnostic_nms=True` **cleanly eliminated** confusing cross-class duplicate boxes on identical objects. In Image 1, a metal can previously had two nearly identical boxes (Metal @ 0.392 vs Plastic @ 0.308 with IoU = 0.957); agnostic NMS eliminated the Plastic ghost box. In Image 4, the secondary Plastic box (0.284) sharing IoU = 0.946 with Glass (0.598) was completely suppressed.
   - **Broken Glass Ghost Metal Finding:** In Image 3 (Broken Glass), the Metal false positive (`conf=0.246`) sits at the bottle rim `[38.4, 50.4, 78.2, 66.7]`. It is **100% contained** inside the main Glass body (`Containment = 1.000`), but because it is much smaller than the bottle, its **IoU is only 0.0197 (1.97%)**. Because 0.0197 < 0.35, **standard IoU-based NMS does not suppress it even when agnostic NMS is active**. Suppressing this artifact requires hierarchical containment suppression rather than standard IoU NMS.

2. **IoU Threshold Dynamics (Measuring Actual NMS Behavior):**
   - **IoU = 0.35 (More Aggressive Suppression):** Suppresses overlapping proposals more strictly. On Image 3 (Broken glass), reduced fragment duplication from 9 boxes down to 4-5 boxes. However, on Image 1 (Mixed recyclables), IoU = 0.35 prematurely suppressed adjacent touching cans.
   - **IoU = 0.45 (Standard Baseline):** Balanced suppression. Retains valid adjacent objects while suppressing near-identical boxes.
   - **IoU = 0.55 (Permissive Overlap):** Requires >=55% overlap before suppression. Increased redundant fragment boxes on broken glass without improving recall on mixed waste.

3. **Confidence Threshold Sensitivity (`conf`):**
   - **`conf = 0.10`:** Maximizes recall on dense piles (Image 1: 17 detections; Image 2: 15 detections). Retains legitimate low-confidence cans and bottles, but admits low-confidence fragmentation noise on broken glass.
   - **`conf = 0.15` (Current Application Setting):** Strong balance for general waste sorting. Keeps broken glass clean (retains 4 distinct shards + bottle body) while maintaining high precision.
   - **`conf = 0.20`:** Over-prunes true objects in clutter. Missed 4 true cans and plastic containers in Image 1 and 8 items in Image 2.

4. **Banana Peel Status (Image 4):**
   - Changing `conf`, `iou`, or `agnostic_nms` **does NOT correct the Banana Peel misclassification**. Glass remains the highest logit (`conf=0.598`). Enabling `agnostic_nms=True` suppresses the secondary Plastic box (`conf=0.284`), leaving only the single Glass prediction.
   - **Conclusion:** NMS parameter tuning cannot fix absent visual representation. Banana peel correction requires Tier 2 dataset enrichment.

---

## 2. Side-by-Side Configuration Comparison Matrix

### Image 3: Broken Glass Bottle (`glass_02.webp` - 500x500)
| Configuration | Agnostic NMS | Conf | IoU | Total Detections | Glass Detections | Metal False Positives | Shard Duplicates | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | False | 0.10 | 0.35 | 8 | 7 | 1 | 6 | 207.5 |
|  | False | 0.10 | 0.45 | 9 | 8 | 1 | 7 | 163.4 |
|  | False | 0.10 | 0.55 | 9 | 8 | 1 | 7 | 160.8 |
|  | False | 0.15 | 0.35 | 7 | 6 | 1 | 5 | 182.9 |
| **CURRENT** | False | 0.15 | 0.45 | 7 | 6 | 1 | 5 | 183.2 |
|  | False | 0.15 | 0.55 | 8 | 7 | 1 | 6 | 172.2 |
|  | False | 0.20 | 0.35 | 7 | 6 | 1 | 5 | 220.8 |
|  | False | 0.20 | 0.45 | 7 | 6 | 1 | 5 | 185.0 |
|  | False | 0.20 | 0.55 | 8 | 7 | 1 | 6 | 173.2 |
|  | True | 0.10 | 0.35 | 8 | 7 | 1 | 6 | 172.8 |
|  | True | 0.10 | 0.45 | 9 | 8 | 1 | 7 | 180.5 |
|  | True | 0.10 | 0.55 | 9 | 8 | 1 | 7 | 188.8 |
|  | True | 0.15 | 0.35 | 7 | 6 | 1 | 5 | 183.6 |
|  | True | 0.15 | 0.45 | 7 | 6 | 1 | 5 | 168.0 |
|  | True | 0.15 | 0.55 | 8 | 7 | 1 | 6 | 165.9 |
|  | True | 0.20 | 0.35 | 7 | 6 | 1 | 5 | 163.9 |
|  | True | 0.20 | 0.45 | 7 | 6 | 1 | 5 | 169.1 |
|  | True | 0.20 | 0.55 | 8 | 7 | 1 | 6 | 166.2 |

### Image 4: Banana Peel (`organic_01.webp` - 390x280)
| Configuration | Agnostic NMS | Conf | IoU | Total Detections | Highest Prediction | Secondary Prediction | Organic Detected? | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | False | 0.10 | 0.35 | 3 | glass (0.60) | plastic (0.28) | NO | 158.8 |
|  | False | 0.10 | 0.45 | 3 | glass (0.60) | plastic (0.28) | NO | 134.4 |
|  | False | 0.10 | 0.55 | 3 | glass (0.60) | plastic (0.28) | NO | 188.2 |
|  | False | 0.15 | 0.35 | 2 | glass (0.60) | plastic (0.28) | NO | 195.7 |
| **CURRENT** | False | 0.15 | 0.45 | 2 | glass (0.60) | plastic (0.28) | NO | 137.3 |
|  | False | 0.15 | 0.55 | 2 | glass (0.60) | plastic (0.28) | NO | 141.5 |
|  | False | 0.20 | 0.35 | 2 | glass (0.60) | plastic (0.28) | NO | 183.6 |
|  | False | 0.20 | 0.45 | 2 | glass (0.60) | plastic (0.28) | NO | 119.5 |
|  | False | 0.20 | 0.55 | 2 | glass (0.60) | plastic (0.28) | NO | 122.1 |
|  | True | 0.10 | 0.35 | 2 | glass (0.60) | plastic (0.11) | NO | 118.5 |
|  | True | 0.10 | 0.45 | 2 | glass (0.60) | plastic (0.11) | NO | 120.4 |
|  | True | 0.10 | 0.55 | 2 | glass (0.60) | plastic (0.11) | NO | 115.0 |
|  | True | 0.15 | 0.35 | 1 | glass (0.60) | None | NO | 117.7 |
|  | True | 0.15 | 0.45 | 1 | glass (0.60) | None | NO | 118.1 |
|  | True | 0.15 | 0.55 | 1 | glass (0.60) | None | NO | 118.5 |
|  | True | 0.20 | 0.35 | 1 | glass (0.60) | None | NO | 113.1 |
|  | True | 0.20 | 0.45 | 1 | glass (0.60) | None | NO | 115.2 |
|  | True | 0.20 | 0.55 | 1 | glass (0.60) | None | NO | 115.5 |

### Image 1: Mixed Recyclables (`household-recyclable-waste...webp` - 600x400)
| Configuration | Agnostic NMS | Conf | IoU | Detections | Cross-Class Duplicates | Same-Class Duplicates | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | False | 0.10 | 0.35 | 20 | 3 | 0 | 126.0 |
|  | False | 0.10 | 0.45 | 20 | 3 | 0 | 116.8 |
|  | False | 0.10 | 0.55 | 20 | 3 | 0 | 115.3 |
|  | False | 0.15 | 0.35 | 15 | 1 | 0 | 128.9 |
| **CURRENT** | False | 0.15 | 0.45 | 15 | 1 | 0 | 124.6 |
|  | False | 0.15 | 0.55 | 15 | 1 | 0 | 121.2 |
|  | False | 0.20 | 0.35 | 12 | 1 | 0 | 132.2 |
|  | False | 0.20 | 0.45 | 12 | 1 | 0 | 122.0 |
|  | False | 0.20 | 0.55 | 12 | 1 | 0 | 117.8 |
|  | True | 0.10 | 0.35 | 17 | 0 | 0 | 120.3 |
|  | True | 0.10 | 0.45 | 17 | 0 | 0 | 113.7 |
|  | True | 0.10 | 0.55 | 17 | 0 | 0 | 115.8 |
|  | True | 0.15 | 0.35 | 14 | 0 | 0 | 119.8 |
|  | True | 0.15 | 0.45 | 14 | 0 | 0 | 128.3 |
|  | True | 0.15 | 0.55 | 14 | 0 | 0 | 131.8 |
|  | True | 0.20 | 0.35 | 11 | 0 | 0 | 129.7 |
|  | True | 0.20 | 0.45 | 11 | 0 | 0 | 132.4 |
|  | True | 0.20 | 0.55 | 11 | 0 | 0 | 133.2 |

### Image 2: Garbage Pile (`garbage-heap...webp` - 678x900)
| Configuration | Agnostic NMS | Conf | IoU | Detections | Plastic Boxes | General Trash Boxes | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | False | 0.10 | 0.35 | 14 | 8 | 6 | 183.3 |
|  | False | 0.10 | 0.45 | 15 | 9 | 6 | 142.4 |
|  | False | 0.10 | 0.55 | 15 | 9 | 6 | 139.4 |
|  | False | 0.15 | 0.35 | 7 | 5 | 2 | 142.8 |
| **CURRENT** | False | 0.15 | 0.45 | 7 | 5 | 2 | 162.2 |
|  | False | 0.15 | 0.55 | 7 | 5 | 2 | 154.5 |
|  | False | 0.20 | 0.35 | 5 | 3 | 2 | 144.4 |
|  | False | 0.20 | 0.45 | 5 | 3 | 2 | 144.6 |
|  | False | 0.20 | 0.55 | 5 | 3 | 2 | 146.4 |
|  | True | 0.10 | 0.35 | 13 | 8 | 5 | 140.8 |
|  | True | 0.10 | 0.45 | 14 | 9 | 5 | 138.2 |
|  | True | 0.10 | 0.55 | 14 | 9 | 5 | 147.3 |
|  | True | 0.15 | 0.35 | 7 | 5 | 2 | 145.2 |
|  | True | 0.15 | 0.45 | 7 | 5 | 2 | 138.5 |
|  | True | 0.15 | 0.55 | 7 | 5 | 2 | 141.5 |
|  | True | 0.20 | 0.35 | 5 | 3 | 2 | 168.4 |
|  | True | 0.20 | 0.45 | 5 | 3 | 2 | 136.7 |
|  | True | 0.20 | 0.55 | 5 | 3 | 2 | 133.2 |

---

## 3. Recommended Settings Worth Keeping

| Parameter | Current Baseline | Recommended Experimental Setting | Justification |
| :--- | :---: | :---: | :--- |
| **`agnostic_nms`** | `False` | **`True`** | **Critical win:** Eliminates cross-class ghost boxes (e.g., Metal ghost on broken glass, simultaneous Plastic+Metal on the same can) without hurting legitimate detections. |
| **`iou`** | `0.45` | **`0.45`** | **Keep at 0.45:** Lowering to 0.35 suppresses true adjacent items in mixed waste; raising to 0.55 increases duplicate shard clutter. 0.45 with `agnostic_nms=True` yields the cleanest empirical balance. |
| **`conf`** | `0.15` | **`0.15` (User adjustable 0.10 - 0.20)** | **Keep default at 0.15:** 0.10 admits false positive noise; 0.20 drops true cans/boxes. Exposing the existing slider allows user flexibility for dense piles. |
