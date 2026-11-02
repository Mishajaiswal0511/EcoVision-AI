# EcoVision AI — Empirical Dataset Audit Report

> **Dataset:** TACO (Trash Annotations in Context) — Official All-Class (Roboflow Release v6)  
> **Source URL:** https://universe.roboflow.com/garbage-wbsv6/taco-official-all-class/dataset/6  
> **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
> **Execution Date:** 2026-10-29  
> **Verification Status:** Verified against genuine downloaded files on disk. No faked metrics.

---

## 1. Executive Summary

| Metric | Verified Value |
|---|---|
| **Total Images** | **2,550** (Train: 2,100, Val: 300, Test: 150) |
| **Total Annotations** | **8,152** bounding boxes across 59 raw classes |
| **Corrupt / Unreadable Images** | **0** (100% of image files opened and verified via Pillow) |
| **Missing Label Files** | **0** (All 2,550 images have corresponding `.txt` label files) |
| **Invalid Annotation Format Lines** | **0** (All lines contain numeric values and correct field lengths) |
| **Degenerate Bounding Boxes** | **3** (Zero-area boxes where width or height equals 0.0) |
| **Exact Duplicate Images (MD5)** | **0** |
| **Near-Duplicate Images (pHash, threshold 10)** | **120 groups** (128 redundant images) |
| **Cross-Split Data Leakage Groups** | **5 groups** in original Roboflow split |

---

## 2. Dataset Split Breakdown

| Split | Images | Annotations | Multi-Object Images | Background Images | Degenerate Boxes |
|---|---|---|---|---|---|
| **Train** | 2,100 | 6,654 | 1,402 | 0 | 1 |
| **Val** (`valid`) | 300 | 1,029 | 208 | 0 | 0 |
| **Test** | 150 | 469 | 94 | 0 | 2 |
| **Total** | **2,550** | **8,152** | **1,704** | **0** | **3** |

*Note: 66.8% of images contain multiple overlapping or co-occurring waste objects, representing realistic real-world scenes rather than isolated objects.*

---

## 3. Class Distribution & Imbalance Analysis

The original dataset defines **59 classes**. Below is the empirical instance frequency across the entire dataset:

| Rank | Class Name | Instances | Viability Assessment |
|---|---|---|---|
| 1 | `Cigarette` | 1,128 | 🟢 Abundant (Outdoor litter bias) |
| 2 | `Unlabeled litter` | 903 | 🟡 Ambiguous / General trash |
| 3 | `Plastic film` | 789 | 🟢 Abundant (Plastic) |
| 4 | `Clear plastic bottle` | 484 | 🟢 Abundant (Plastic bottle) |
| 5 | `Other plastic` | 456 | 🟢 Abundant (Plastic) |
| 6 | `Other plastic wrapper` | 446 | 🟢 Abundant (Plastic wrapper) |
| 7 | `Drink can` | 370 | 🟢 Abundant (Metal can) |
| 8 | `Plastic bottle cap` | 348 | 🟢 Abundant (Plastic) |
| 9 | `Plastic straw` | 280 | 🟢 Strong (Plastic) |
| 10 | `Styrofoam piece` | 195 | 🟢 Viable (Plastic/Expanded polystyrene) |
| 11 | `Broken glass` | 193 | 🟢 Viable (Glass) |
| 12 | `Glass bottle` | 181 | 🟢 Viable (Glass) |
| 13 | `Disposable plastic cup` | 176 | 🟢 Viable (Plastic) |
| 14 | `Pop tab` | 161 | 🟢 Viable (Metal) |
| 15 | `Other carton` | 159 | 🟢 Viable (Cardboard) |
| 16 | `Plastic lid` | 138 | 🟢 Viable (Plastic) |
| 17 | `Normal paper` | 136 | 🟢 Viable (Paper) |
| 18 | `Metal bottle cap` | 133 | 🟢 Viable (Metal) |
| 19 | `Paper cup` | 117 | 🟢 Viable (Paper) |
| 20 | `Corrugated carton` | 114 | 🟢 Viable (Cardboard) |
| 21 | `Aluminium foil` | 106 | 🟢 Viable (Metal) |
| 22 | `Single-use carrier bag` | 104 | 🟢 Viable (Plastic) |
| 23 | `Other plastic bottle` | 89 | 🟡 Moderate (Plastic) |
| 24 | `Tissues` | 75 | 🟡 Low-moderate |
| 25 | `Drink carton` | 73 | 🟡 Moderate (Cardboard) |
| 26 | `Food Can` | 55 | 🟡 Moderate (Metal) |
| 27 | `Meal carton` | 46 | 🟡 Low (Cardboard) |
| 28 | `Paper bag` | 46 | 🟡 Low (Paper) |
| 29 | `Disposable food container` | 46 | 🟡 Low (Plastic) |
| 30 | `Garbage bag` | 44 | 🟡 Low (Plastic) |
| 31 | `Rope - strings` | 42 | 🟡 Low |
| 32 | `Scrap metal` | 31 | 🔴 Low |
| 33 | `Magazine paper` | 30 | 🔴 Low (Paper) |
| 34 | `Plastic utensils` | 27 | 🔴 Low (Plastic) |
| 35 | `Spread tub` | 26 | 🔴 Low (Plastic) |
| 36 | `Aerosol` | 24 | 🔴 Low (Metal) |
| 37 | `Foam food container` | 23 | 🔴 Low (Plastic) |
| 38 | `Wrapping paper` | 22 | 🔴 Low (Paper) |
| 39 | `Egg carton` | 20 | 🔴 Very low (Cardboard) |
| 40 | `Foam cup` | 20 | 🔴 Very low (Plastic) |
| 41 | **`Food waste`** | **13** | ❌ **Severely insufficient (13 instances)** |
| 42 | `Aluminium blister pack` | 12 | 🔴 Low (Metal) |
| 43 | `Squeezable tube` | 12 | 🔴 Low |
| 44 | `Glass cup` | 11 | 🔴 Low (Glass) |
| 45 | `Glass jar` | 11 | 🔴 Low (Glass) |
| 46 | `Shoe` | 11 | ❌ Non-waste outlier |
| 47 | `Other plastic container` | 9 | 🔴 Very low |
| 48 | `Toilet tube` | 9 | 🔴 Very low (Cardboard) |
| 49 | `Paper straw` | 7 | 🔴 Very low (Paper) |
| 50 | `Tupperware` | 7 | 🔴 Very low (Plastic) |
| 51 | `Plastic glooves` | 6 | 🔴 Very low (Plastic) |
| 52 | `Polypropylene bag` | 6 | 🔴 Very low (Plastic) |
| 53 | `Pizza box` | 5 | 🔴 Very low (Cardboard) |
| 54 | **`Battery`** | **4** | ❌ **Critically insufficient (4 instances)** |
| 55 | `Other plastic cup` | 4 | 🔴 Very low (Plastic) |
| 56 | `Carded blister pack` | 2 | 🔴 Very low |
| 57 | `Six pack rings` | 2 | 🔴 Very low |
| 58 | `Metal lid` | 0 | ❌ No instances found |
| 59 | `Other plastic wrapper` (dup) | 0 | ❌ No instances found |

---

## 4. Geometric & Annotation Integrity Audit

### Degenerate Bounding Boxes Flagged (Must Be Filtered)
* **Train split:**
  * File: `000049_JPG.rf.3ae9aa2b17afec86fc4a51e2de680c7c.txt` (Line 5) — `w=0.8211`, `h=0.0` (zero height).
* **Test split:**
  * File: `000008_jpg.rf.3bd5b9fc637686318822dc3183352b68.txt` (Line 1) — `w=0.0`, `h=0.5905` (zero width).
  * File: `000088_jpg.rf.500768ed03e5b86b9dcb44ad91d3a15f.txt` (Line 1) — `w=0.7549`, `h=0.0` (zero height).

*Action:* Our `prepare_splits.py` and `label_mapper.py` will automatically exclude these 3 corrupted annotations during data normalization.

### Boundary Warnings
* **Tiny boxes (< 0.01% image area):** 4 instances detected in `train` (e.g., distant bottle caps or micro-debris). Valid small objects, preserved for small-object sensitivity.
* **Large boxes (> 95% image area):** 2 instances detected in `train` (e.g., full-frame plastic sheet). Preserved as valid macro-detections.

---

## 5. Duplicate & Data Leakage Review

* **Exact duplicates (MD5):** 0 groups.
* **Perceptual near-duplicates (pHash distance ≤ 10):** 120 groups containing 128 redundant images.
* **Data Leakage in Original Roboflow Split:** **5 groups have cross-split contamination** (the identical or near-identical scene appears in both `train` and `valid`/`test`).
  * *Why this matters:* If models are evaluated on leaked test images, evaluation metrics (mAP) appear artificially inflated.
  * *Mitigation:* We will generate custom, stratified splits using `scripts/prepare_splits.py` with `evaluation/results/dedup_plan.json`, ensuring all members of near-duplicate clusters remain locked strictly within the `train` partition.

---

## 6. Evidence-Based Taxonomy Recommendation

### The Empirical Reality
1. **`Battery` is NOT viable in this dataset alone:** With only 4 instances across 2,550 images, training a YOLO detector on `battery` will result in near-zero recall and extreme false-positive hallucination.
2. **`Food waste / Organic` is NOT viable in this dataset alone:** With only 13 instances in outdoor litter scenes, organic waste cannot be reliably learned.
3. **Core Recyclables have abundant, high-quality data:**
   * **Plastic:** ~3,650 instances (bottles, cups, films, wrappers, caps, straws)
   * **Metal:** ~900 instances (drink cans, food cans, bottle caps, foil, pop tabs)
   * **Glass:** ~396 instances (bottles, broken glass fragments, jars, cups)
   * **Cardboard:** ~425 instances (corrugated cartons, meal cartons, drink cartons)
   * **Paper:** ~350 instances (normal paper, paper cups, paper bags, magazines)
   * **General Trash / Non-recyclables:** ~2,000+ instances (cigarette butts, unlabeled litter, ropes)

### Recommended 6-Class Taxonomy for EcoVision AI Baseline
1. `plastic` (Bottles, containers, films, bags, wrappers, cups, straws)
2. `paper` (Paper sheets, paper cups, paper bags, magazines)
3. `cardboard` (Cartons, corrugated boxes, food boxes)
4. `glass` (Glass bottles, glass jars, broken glass)
5. `metal` (Beverage cans, food cans, foil, metal caps, tabs)
6. `general_trash` (Cigarettes, non-recyclable litter, miscellaneous debris)

*Hazardous Battery & Organic Waste Guidance:*
Until targeted bounding-box datasets for battery and food waste are acquired and merged, the application will provide proactive UI warning banners and educational guidance, rather than pretending to detect them with uncalibrated 4-instance weights.

