# EcoVision AI — Tier 2 Proposal: V5.1 Dataset Enrichment & Retraining

**Date:** November 1, 2026  
**Status:** PROPOSAL ONLY (Zero training initiated; strictly awaiting user sign-off)  
**Target Release:** EcoVision AI V5.1 (Robust Organic & General Trash Detection)  
**Author:** Antigravity Senior Computer Vision Engineer  

---

## 1. Executive Summary & Problem Statement

The Tier 1 controlled inference experiment proved that post-processing parameters (`conf`, `iou`, `agnostic_nms`) **cannot solve visual data starvation**:
- The banana peel was predicted as **Glass (60% conf)** and **Plastic (28% conf)** because the model has essentially zero visual training signal for standalone macro food waste.
- In V5 training, Class 6 (`organic`) was confined to only **92 images (1.1% of the dataset)**, with over 50% of instances concentrated in just 10 compost heap photos. Standalone fruit peels, vegetables, and household wet waste were completely absent.
- Class 5 (`general_trash`) achieved only **14.3% validation recall**, causing 85% of residual items in garbage piles to be missed.

**Tier 2 Objective:** Assemble an enriched, balanced, parent-scene grouped dataset (`data/processed_v5_1/`) that provides diverse visual supervision for standalone organic food waste and distinct non-recyclable residual trash, followed by a controlled training run of YOLO11s at 800×800.

---

## 2. Source Ingestion & Target Volumes

### A. Standalone Organic Waste Enrichment (Class 6: `organic`)
We will tap the 45,395 annotations from `mcl2/general-waste-data/1` category `BIODEGRADABLE` that were deliberately excluded during V3 construction, supplemented by open CC-licensed food waste benchmarks:

| Target Sub-Category | Visual Representation | Target Training Instances | Target Validation Instances | Source Pool |
| :--- | :--- | :---: | :---: | :--- |
| **Fruit Peels & Cores** | Banana peels, orange rinds, melon rinds, apple cores | 650 | 180 | GWD `BIODEGRADABLE` + TACO Food Waste |
| **Vegetable Trimmings** | Potato peels, onion skins, carrot ends, greens | 550 | 150 | GWD `BIODEGRADABLE` |
| **Household Food Leftovers** | Cooked rice, bread crusts, meal residues, eggshells | 500 | 140 | GWD `BIODEGRADABLE` |
| **Coffee/Tea Grounds** | Spent tea leaves, espresso pucks, filter grounds | 300 | 80 | Curated open food waste |
| **Macro Organic Target Total** | — | **2,000** | **550** | — |

*Projected V5.1 Organic Totals:* ~3,500 train instances across 1,000+ unique images (increasing organic image representation from 1.1% to >10% of the dataset).

### B. Residual & Soiled Packaging Enrichment (Class 5: `general_trash`)
To resolve the 14.3% recall bottleneck and eliminate visual confusion with clean plastic:

| Target Sub-Category | Visual Representation | Target Training Instances | Disambiguation Supervision |
| :--- | :--- | :---: | :--- |
| **Multilayer Snack Packets** | Metallized chip/biscuit bags (BOPP film) | +400 | Teaches reflective crinkled foil film != recyclable PET |
| **Food-Soiled Paperboard** | Greasy pizza boxes, curry-stained meal cartons | +350 | Teaches grease stains != clean cardboard |
| **Expanded Polystyrene** | Styrofoam cups, clamshell containers, packaging beads | +300 | Teaches beaded white foam != rigid plastic |
| **Sanitary & Hygiene Items** | Used tissues, wipes, cleaning gloves (marked with special care) | +250 | High-contrast sanitary supervision |
| **Plastic Straws & Cutlery** | Single-use black/colored straws, plastic spoons/forks | +300 | Teaches banned single-use plastics = general trash |
| **General Trash Target Expansion** | — | **+1,600** | Increases train instances from 1,926 to ~3,500 |

---

## 3. Strict Safety & Verification Safeguards

1. **Production & Previous Checkpoints Sealed:**
   - `models/weights_best.pt` remains 100% untouched (SHA-256: `1933635e...`).
   - Candidate checkpoint `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt` remains untouched as a baseline.
2. **Evaluation Split Isolation:**
   - `data/real_world_eval/` remains **completely sealed and untouched**.
3. **Parent-Scene Leakage Verification:**
   - All newly ingested images will be grouped by parent-scene hash using the existing `verify_dataset_leakage.py` pipeline.
   - **Zero parent-scene overlap (0.00% contamination)** between train and val will be enforced before training is permitted.
4. **Isolated Storage:**
   - Dataset will be created in `data/processed_v5_1/`.
   - Training artifacts and weights will be stored exclusively under `models/experiments/v5_1_enrichment/`.

---

## 4. Planned Training Configuration (YOLO11s V5.1)

| Parameter | Planned Setting | Rationale |
| :--- | :---: | :--- |
| **Architecture** | `yolo11s.pt` | Retains AMD Ryzen 7 7735U real-time CPU latency (~171 ms @ 800px) |
| **Input Resolution** | `imgsz=800` | Essential for small packaging fragments and litter |
| **Optimizer & LR** | `AdamW`, `lr0=0.001` | Proven stable convergence in V5 experiment |
| **Epochs** | 50 epochs (patience 15) | Sufficient convergence with early stopping |
| **Augmentation** | `mosaic=1.0`, `mixup=0.15`, `degrees=10.0`, `fliplr=0.5` | Standardizes rotation invariance for dropped fruit peels |
| **Loss Weights** | Class loss weight `cls=1.2` | Boosts penalization for under-represented classes |

---

## 5. Success Acceptance Criteria

| Metric | V5 Baseline | V5.1 Target Threshold | Verification Test |
| :--- | :---: | :---: | :--- |
| **Organic Recall** | 32.2% | **$\ge 65.0\%$** | Approved V5.1 validation split |
| **Organic mAP@50** | 33.3% | **$\ge 60.0\%$** | Approved V5.1 validation split |
| **General Trash Recall**| 14.3% | **$\ge 40.0\%$** | Approved V5.1 validation split |
| **Overall mAP@50** | 52.0% | **$\ge 58.0\%$** | Approved V5.1 validation split |
| **Banana Peel Test** | Glass (0.60) / Plastic (0.28) | **Organic $\ge 0.70$ (Zero Glass/Plastic)** | `organic_01.webp` user test image |
| **Broken Glass Test** | 1 Metal ghost box | **0 Metal ghost boxes (Pure Glass)** | `glass_02.webp` user test image |

---

> [!IMPORTANT]
> **Status: Paused for Approval.**  
> No images have been downloaded, no datasets modified, and no training processes launched. Training will only proceed upon your explicit direction.

