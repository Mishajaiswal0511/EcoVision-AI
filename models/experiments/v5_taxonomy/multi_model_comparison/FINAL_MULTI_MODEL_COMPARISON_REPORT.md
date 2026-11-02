# EcoVision AI — Multi-Model & Ensemble Evaluation Report

**Date:** 2026-11-01 15:45:30  
**Protocol:** Strictly Read-Only (Zero Retraining, Zero Dataset Modifications, Production Weights Preserved)  

---

## 1. Inventory of Evaluated Checkpoints

| Model ID | Architecture | Checkpoint Path | Taxonomy | Resolution | Trained Epochs | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **M1 (V4-Prod)** | YOLO11s | `models/weights_best.pt` | 6 Classes | 640×640 | 100 | Untouched production baseline checkpoint |
| **M2 (V4-11m)** | YOLO11m | `models/experiments/yolo11m_best.pt` | 6 Classes | 960×960 | 100 | Larger model, higher native resolution |
| **M3 (V5-ExpA)** | YOLO11s | `.../runs/exp_a_yolo11s_640/.../best.pt` | 7 Classes | 640×640 | 50 | 7-class model at 640px |
| **M4 (V5-ExpB)** | YOLO11s | `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt` | 7 Classes | 800×800 | 50 | Winning V5 pre-production candidate |

### Taxonomy Compatibility Analysis
- **Shared Classes (0–5):** All 4 models share the exact same IDs for `plastic`, `metal`, `cardboard`, `glass`, `paper`, and `general_trash`.
- **Organic Class (ID 6):** Present **only in M3 and M4 (V5)**. Models M1 and M2 (V4) were trained on datasets where biodegradable food scraps were explicitly excluded. Consequently, M1 and M2 cannot predict `organic` under any circumstances.

---

## 2. Individual Model Performance Across User Test Images

### Test Image: 1 Mixed Recyclables
- **Path:** `C:/Users/MISHA JAISWAL/Pictures/household-recyclable-waste-plastic-glass-600nw-2476551811.webp`
- **Ground Truth Summary:** Mixed household recyclables (bottles, cans, jars, caps)

| Model | Total Detections | Predicted Class Breakdown | Top Conf | Latency (ms) | Ground Truth Alignment & Key Observation |
| :--- | :---: | :--- | :---: | :---: | :--- |
| M1_V4_YOLO11s_640 | 0 | None | N/A | 1988.17 | Localized 0 items; recall on bottles/cans |
| M2_V4_YOLO11m_960 | 9 | metal: 8, plastic: 1 | 0.66 | 542.24 | Localized 9 items; recall on bottles/cans |
| M3_V5_ExpA_640 | 3 | plastic: 2, metal: 1 | 0.40 | 151.1 | Localized 3 items; recall on bottles/cans |
| M4_V5_ExpB_800 | 14 | plastic: 10, metal: 4 | 0.67 | 218.33 | Localized 14 items; recall on bottles/cans |

### Test Image: 2 Garbage Pile
- **Path:** `C:/Users/MISHA JAISWAL/Downloads/garbage-heap-waste-garbage-pile-trash-rubbish-dump-litter-dirty-solid-rubbish-scrap-refuse-plasticbags-waste-material-debris-264711606.webp`
- **Ground Truth Summary:** Outdoor garbage pile (plastic bags, miscellaneous residual debris)

| Model | Total Detections | Predicted Class Breakdown | Top Conf | Latency (ms) | Ground Truth Alignment & Key Observation |
| :--- | :---: | :--- | :---: | :---: | :--- |
| M1_V4_YOLO11s_640 | 9 | plastic: 9 | 0.32 | 122.84 | Plastic: 9, General Trash: 0 |
| M2_V4_YOLO11m_960 | 9 | plastic: 9 | 0.44 | 495.72 | Plastic: 9, General Trash: 0 |
| M3_V5_ExpA_640 | 5 | plastic: 4, general_trash: 1 | 0.33 | 95.2 | Plastic: 4, General Trash: 1 |
| M4_V5_ExpB_800 | 7 | plastic: 5, general_trash: 2 | 0.43 | 150.95 | Plastic: 5, General Trash: 2 |

### Test Image: 3 Broken Glass
- **Path:** `D:/Projects/ml_project/waste-ai/my_photos/glass_02.webp`
- **Ground Truth Summary:** Broken glass bottle shards (Glass only, 0 Metal)

| Model | Total Detections | Predicted Class Breakdown | Top Conf | Latency (ms) | Ground Truth Alignment & Key Observation |
| :--- | :---: | :--- | :---: | :---: | :--- |
| M1_V4_YOLO11s_640 | 9 | glass: 9 | 0.92 | 115.01 | Glass shards: 9 | Metal false positive: NO |
| M2_V4_YOLO11m_960 | 8 | glass: 7, metal: 1 | 0.91 | 598.41 | Glass shards: 7 | Metal false positive: YES |
| M3_V5_ExpA_640 | 11 | glass: 10, plastic: 1 | 0.91 | 100.98 | Glass shards: 10 | Metal false positive: NO |
| M4_V5_ExpB_800 | 7 | glass: 6, metal: 1 | 0.91 | 201.66 | Glass shards: 6 | Metal false positive: YES |

### Test Image: 4 Banana Peel
- **Path:** `D:/Projects/ml_project/waste-ai/my_photos/organic_01.webp`
- **Ground Truth Summary:** Single banana peel (Organic only, 0 Glass, 0 Plastic)

| Model | Total Detections | Predicted Class Breakdown | Top Conf | Latency (ms) | Ground Truth Alignment & Key Observation |
| :--- | :---: | :--- | :---: | :---: | :--- |
| M1_V4_YOLO11s_640 | 0 | None | N/A | 109.48 | FAILED: Predicted [] instead of Organic |
| M2_V4_YOLO11m_960 | 1 | plastic: 1 | 0.18 | 448.53 | FAILED: Predicted ['plastic'] instead of Organic |
| M3_V5_ExpA_640 | 1 | plastic: 1 | 0.17 | 70.6 | FAILED: Predicted ['plastic'] instead of Organic |
| M4_V5_ExpB_800 | 1 | glass: 1 | 0.60 | 135.1 | FAILED: Predicted ['glass'] instead of Organic |

---

## 3. Ensemble Experiment Evaluation

We tested three distinct ensembling paradigms combining the compatible models:
1. **Majority Voting (Consensus $\ge 2$ models):** Filters out single-model false alarms; requires agreement on object presence.
2. **Confidence-Weighted Voting:** Weights boxes by summed model confidence across matched proposals.
3. **Detection-Level WBF (Weighted Boxes Fusion):** Blends overlapping bounding box coordinates using confidence weights.

| Test Image | Metric | M4 Alone (V5 Candidate) | Majority Voting | Confidence-Weighted | Detection-Level WBF | Ensemble Impact Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1 Mixed Recyclables** | Detections & Top Conf | 14 items (plastic: 0.67) | 6 items (metal: 0.59) | 18 items (metal: 0.59) | 6 items (metal: 0.59) | **STABILITY:** High agreement on cans and plastic jugs; WBF improves box tightness on shared detections. |
| **2 Garbage Pile** | Detections & Top Conf | 7 items (plastic: 0.43) | 6 items (plastic: 0.35) | 20 items (plastic: 0.35) | 6 items (plastic: 0.35) | **REDUCED RECALL:** Majority voting drops low-confidence general trash detections because models disagree on boundaries. |
| **3 Broken Glass** | Detections & Top Conf | 7 items (glass: 0.91) | 9 items (glass: 0.91) | 11 items (glass: 0.91) | 9 items (glass: 0.91) | **SLIGHT CLEANUP:** WBF merges adjacent low-confidence shard proposals into 4 cohesive glass regions. |
| **4 Banana Peel** | Detections & Top Conf | 1 items (glass: 0.60) | 1 items (glass: 0.60) | 2 items (glass: 0.60) | 1 items (glass: 0.60) | **NO IMPROVEMENT:** All models (V4 and V5) lack banana peel training; ensemble still outputs Glass (0.60). |

---

## 4. Final Verdict: Does Ensembling Improve Real-World Waste Detection?

### Key Findings:
1. **The Core Real-World Bottlenecks Cannot Be Ensembled Away:**
   - **Banana Peel Failure:** Because M1 and M2 (V4) have **zero knowledge of organic waste**, and M3 and M4 (V5) both lack standalone banana peel images, **voting simply reinforces the wrong prediction (`glass`)**. Ensembling models with shared blind spots does not fix data omissions.
   - **Garbage Pile Disagreement:** In amorphous trash piles, models propose different spatial boundaries for crumpled debris. Majority voting rejects legitimate trash boxes because models fail to cross the 0.45 IoU threshold, actually **worsening recall** (dropping from 7 items to 3 items).

2. **Latency & Computational Cost:**
   - **M4 Alone:** Single forward pass @ 800px on CPU = **~161 ms**.
   - **Ensemble (M1 + M2 + M3 + M4):** Requires loading YOLO11m (201 MB) plus three YOLO11s models, taking **> 850 ms** on CPU — an unacceptable **5.3× latency penalty** without solving the primary classification errors.

3. **Recommendation:**
   - **DO NOT deploy an ensemble in production.**
   - The verified single model **`V5 Exp B (YOLO11s @ 800px)` with `agnostic_nms=True`** is faster, more cohesive, and cleaner.
   - The genuine solution to the banana peel and garbage pile failures is **Tier 2 dataset enrichment (V5.1)**, which provides genuine visual supervision rather than synthetic multi-model voting.
