# EcoVision AI — Model Failure Audit & Next-Generation Dataset Improvement Plan

**Document Version:** 1.0  
**Date:** October 29, 2026  
**Auditor / Lead Engineer:** Senior Computer Vision Engineer & ML Architect  
**Project Workspace:** `D:\Projects\ml_project\472`  
**Current Dataset:** `data/processed_bbox_v2/` (2,543 images, 8,122 boxes, 6 classes)  
**Target Next Dataset:** `data/processed_v3/` (Proposed)  

---

## 1. Executive Summary & Audit Overview

Following the successful correction of the polygon-to-bounding-box annotation bug (documented in [`docs/label_fix_review.md`](file:///d:/Projects/ml_project/472/docs/label_fix_review.md)), the YOLO11s model achieved the following baseline validation metrics:
- **Precision:** 39.7%
- **Recall:** 33.9%
- **mAP@50:** 30.7%
- **mAP@50–95:** 22.7%

While these metrics represent a significant improvement over the corrupt-box baseline (mAP50 11.9%), testing on real-world smartphone photographs revealed **critical failure modes**:
1. Non-waste household items (chargers, lamps, slippers, shoes, toothbrushes) trigger high-confidence waste detections.
2. Cardboard and paper exhibit severe cross-class confusion (e.g. paper notebooks detected as cardboard).
3. Test images exhibit multi-class hallucination (predicting multiple overlapping boxes of different classes on a single object).

This document presents a **forensic failure investigation**, audits candidate public datasets with verified licenses, resolves the non-waste false positive problem, and outlines an evidence-based roadmap for `data/processed_v3`.

---

## 2. Forensic Investigation of Real-World Model Failures

### Failure Case 1: White Charger Adapter Classified as Paper (~41% Confidence)
* **Root Cause (Visual & Inductive Bias):**
  * Phone chargers feature white injection-molded plastic with sharp 90-degree rectilinear geometry, smooth planar surfaces, and matte-to-semi-gloss reflectance.
  * In the training dataset (TACO), the `paper` class is predominantly composed of flat white paper sheets, receipts, meal containers, and white folded cartons.
  * Because the detector was trained with **zero non-waste background images**, the network's classification head is forced to distribute probability strictly among the 6 waste classes. The rectilinear white planar surface matches the low-level convolutional activations of paper.

### Failure Case 2: Table Lamp Classified as Cardboard (76.77% Confidence) with Multi-Class Hallucinations
* **Root Cause (Geometric Prior & Multi-Part Texture):**
  * Table lamps feature cylindrical/conical parchment or fabric shades in beige, tan, or off-white. In TACO, `cardboard` consists of brown corrugated shipping cartons, coffee sleeves, and cardboard cylinders (`Toilet tube`).
  * A table lamp in an indoor photograph is a prominent, salient foreground object. Because the model was never trained to suppress clean indoor objects, the shade activates the `cardboard` feature maps, the metal stalk activates `metal`, and the base activates `plastic`.
  * The confidence of 76.77% demonstrates that confidence thresholding alone cannot prevent false positives.

### Failure Case 3: Shoes, Shoe Racks, and Toothbrushes Receiving Waste Detections
* **Root Cause (The Dropped-Category Vacuum & Material Confusion):**
  * In the raw TACO dataset, `Shoe` was originally class 48 (11 instances). During Step 2.2 remapping, `Shoe` was mapped to `null` and excluded. In the outdoor images where shoes appeared, they were left unannotated, but the model never learned a generalized visual concept of footwear as "non-waste".
  * Toothbrushes are made of injection-molded polypropylene/PET with nylon bristles. Material-wise, they are plastic; context-wise, they are reusable hygiene tools. Without negative examples or context, the model correctly identifies the material as `plastic` but fails to recognize that it is not waste.

### Failure Case 4: Slipper Receiving Low-Confidence `general_trash` Detection
* **Root Cause (Entropy in `general_trash`):**
  * In the current dataset, `general_trash` is heavily dominated by 1,128 `Cigarette` butts and 903 `Unlabeled litter` fragments.
  * "Unlabeled litter" acts as an entropic catch-all for irregular, distorted, or dirty shapes on the ground. A slipper resting on a floor closely matches the spatial prior of discarded floor litter.

### Failure Case 5: Paper Notebook Producing Multiple Cardboard Detections
* **Root Cause (Physical Material Duality):**
  * Spiral and bound notebooks consist of a stiff greyboard or paperboard cover (which is physically cardboard) encasing white paper sheets (which is paper).
  * The model detects the brown/grey stiff cover as `cardboard` and adjacent edges as additional cardboard boxes, resulting in duplicate detections.

### Failure Case 6: Cardboard Test Image Detecting Cardboard + Extraneous Classes
* **Root Cause (Composite Packaging & Anchor Competition):**
  * Commercial cardboard packaging frequently includes transparent packaging tape (`plastic`), glossy printed barcodes (`paper`), and staples (`metal`).
  * Anchor-free detection heads without cross-class suppression fire independently on these sub-regions.

---

## 3. Candidate Dataset Research & Verification

We conducted verified API and web investigations into candidate datasets.

### Candidate 1: TACO (Current Baseline)
- **Source:** Pedro F. Proença et al. / Roboflow Universe `garbage-wbsv6/taco-official-all-class/6`
- **URL:** https://github.com/pedropro/TACO
- **License:** CC BY 4.0 (Verified)
- **Image Count:** 2,543 images (Train: 1,790, Val: 379, Test: 374)
- **Usable Annotations:** 8,122 true bounding boxes
- **Taxonomy:** 59 raw classes mapped to 6 target classes
- **Suitability:** High-fidelity real-world litter in outdoor environments (beaches, streets, parks).
- **Key Limitation:** Severe class starvation for `glass` (396), `paper` (417), and `cardboard` (438). Zero negative/background images.

### Candidate 2: General Waste Data (Roboflow Universe: `mcl2/general-waste-data`)
- **Source:** Roboflow Universe (`mcl2/general-waste-data/1`)
- **URL:** https://universe.roboflow.com/mcl2/general-waste-data
- **License:** CC BY 4.0 (Verified via Roboflow API)
- **Image Count:** **10,464 base images** (Train: 7,324, Val: 2,098, Test: 1,042)
- **Native Resolution:** $640 \times 640$ (No offline rotation or pre-baked distortion in v1)
- **Export Availability:** Pre-generated YOLOv11 export available (~390 MB)
- **Class Breakdown:**
  * `PLASTIC`: 11,132 annotations
  * `GLASS`: 7,812 annotations
  * `METAL`: 5,852 annotations
  * `CARDBOARD`: 4,704 annotations
  * `PAPER`: 4,397 annotations
  * `BIODEGRADABLE`: 45,418 annotations
- **Suitability:** **Outstanding.** It directly solves the critical data starvation in our baseline:
  * Glass instances increase from 396 to $>8,200$ ($20\times$ increase).
  * Cardboard instances increase from 438 to $>5,100$ ($11\times$ increase).
  * Paper instances increase from 417 to $>4,800$ ($11\times$ increase).
  * Metal instances increase from 900 to $>6,700$ ($7\times$ increase).

### Candidate 3: Garbage Classification 3 (`material-identification/garbage-classification-3`)
- **Source:** Roboflow Universe (`material-identification/garbage-classification-3/2`)
- **URL:** https://universe.roboflow.com/material-identification/garbage-classification-3
- **License:** CC BY 4.0 (Verified)
- **Finding:** API query confirmed it is a derivative fork of the same 10,464 base images, but resized to $416 \times 416$ with pre-baked offline 90-degree rotations.
- **Verdict:** **Disqualified in favor of `mcl2/general-waste-data/1`**, which retains the native $640 \times 640$ resolution matching YOLO11s.

### Candidate 4: Underwater & Single-Class Datasets (TrashCan, MJU-Waste)
- **TrashCan 1.0/2.0:** Sourced from deep-sea ROVs; seabed background and marine fauna do not transfer to terrestrial municipal waste.
- **MJU-Waste:** Only provides a single class (`waste`); lacks multi-material segregation capability.

---

## 4. Dataset Compatibility & Class-Mapping Audit

A rigorous semantic mapping matrix is required before merging `mcl2/general-waste-data` with TACO:

| Source Class (`general-waste-data`) | Raw Count | EcoVision Target Class | Target Class ID | Semantic Justification & Precautions |
| :--- | :---: | :---: | :---: | :--- |
| `PLASTIC` | 11,132 | `plastic` | `0` | Direct 1:1 match. Covers PET bottles, HDPE jugs, plastic bags, cups, and trays. |
| `METAL` | 5,852 | `metal` | `1` | Direct 1:1 match. Covers aluminum cans, tin food cans, metal caps, foil, and aerosol containers. |
| `CARDBOARD` | 4,704 | `cardboard` | `2` | Direct 1:1 match. Covers shipping cartons, corrugated boxes, and packaging sleeves. |
| `GLASS` | 7,812 | `glass` | `3` | Direct 1:1 match. Covers beverage bottles, food jars, and glassware. |
| `PAPER` | 4,397 | `paper` | `4` | Direct 1:1 match. Covers clean sheets, newspapers, paper bags, and envelopes. |
| `BIODEGRADABLE` | 45,418 | **EXCLUDE / SAMPLE** | — | **CRITICAL AUDIT DECISION:** Do NOT dump all 45,418 food items into `general_trash`. Doing so would overwhelm the dataset ($>60\%$ food waste), diluting recyclable detection. We propose either excluding it entirely or sampling a small representative set (~1,500 soiled paper/food containers) mapped to `general_trash`. |
| `CLOTH` | 6 | `null` | — | Excluded due to negligible count (6 instances). |

---

## 5. Architectural Solutions for Non-Waste False Positives

To satisfy the critical requirement of distinguishing waste from ordinary household objects, three technical approaches were evaluated:

| Architectural Dimension | Approach 1: Integrated YOLO with Curated Negative Images | Approach 2: Two-Stage Detector + Reusability Classifier | Approach 3: Open-Vocabulary / Vision-Language Filter (CLIP) |
| :--- | :--- | :--- | :--- |
| **Mechanism** | Train YOLO11s on 90% positive waste images + 10% curated negative images (empty `.txt` files). | Stage 1 (YOLO11s) proposes boxes; Stage 2 (MobileNetV4) verifies crop as `waste` vs `reusable`. | YOLO11s proposes boxes; MobileCLIP scores crop against text prompt pairs. |
| **Inference Latency** | **Fastest:** ~2–5 ms on GPU; ~25–40 ms on CPU. | **Fast:** Stage 1 + Stage 2 = ~35–50 ms on CPU. | **Slow:** ~150–350 ms on CPU; high memory footprint. |
| **Model Size** | Single `.pt` model (~19 MB). | Two models (~19 MB + ~10 MB = 29 MB). | Large (>300 MB to 1.5 GB). |
| **Frontend/Backend Impact** | **Zero changes:** Existing REST API and frontend remain 100% compatible. | **Minor:** Chained call in backend prediction service. | **Major:** Heavy Python dependencies (transformers, tokenizers). |
| **Failure Mode Coverage** | Suppresses lamps, chargers, furniture, and floors from triggering detections. | High precision on borderline items (e.g. clean charger vs e-waste charger). | Flexible open-world text queries. |
| **Deployment Recommendation** | **Primary Baseline Requirement (Step 4.1):** Native, zero-latency, essential foundation. | **Secondary Upgrade Option (Step 5):** Deployable if specific borderline items persist. | Not recommended for real-time edge/browser inference. |

### Negative Dataset Formulation
We recommend incorporating **~800 verified negative images** (~8% of the final training split):
- **Household Electronics:** 200 images of chargers, phone plugs, laptops, keyboards, computer mice, power strips.
- **Living Space & Furniture:** 200 images of table lamps, floor lamps, desks, chairs, sofas, bookshelves.
- **Footwear & Apparel:** 200 images of shoes, slippers, sneakers on clean floors, shoe racks, folded bedsheets, clothes.
- **Personal Care & Study:** 200 images of toothbrushes on bathroom vanities, soap dispensers, open/closed notebooks, pens on desks.
- **Verification Rule:** Every negative image must be manually/automatically verified to contain **zero** target waste items and paired with an empty 0-byte `.txt` label file.

---

## 6. Proposed Leakage-Free Dataset Pipeline (`data/processed_v3`)

### Dataset Architecture
The unified `data/processed_v3` dataset will merge:
1. **TACO Cleaned (`data/processed_bbox_v2`):** 2,543 images (outdoor street/beach litter).
2. **General Waste Data (`mcl2/general-waste-data/1`):** 10,464 images (indoor/outdoor segregated recyclables).
3. **Curated Negative Subset:** ~800 images (household objects, electronics, lamps, shoes, notebooks).

### Projected Dataset Statistics (Estimated Post-Merge)
| Class ID | Target Class | Baseline v2 Boxes | General Waste Boxes | Projected Total Annotations |
| :---: | :--- | :---: | :---: | :---: |
| `0` | `plastic` | 3,877 | 11,132 | **~15,000** |
| `1` | `metal` | 900 | 5,852 | **~6,750** |
| `2` | `cardboard` | 438 | 4,704 | **~5,140** |
| `3` | `glass` | 396 | 7,812 | **~8,200** |
| `4` | `paper` | 417 | 4,397 | **~4,810** |
| `5` | `general_trash` | 2,094 | 0 (or ~1,500 sampled) | **~2,100 – 3,600** |
| **Negative** | `[background]` | 0 | 0 | **~800 empty images** |
| **Total** | — | **8,122** | **33,897** | **~42,000+ annotations (~13,800 images)** |

### Leakage-Free Partitioning Strategy
- **Exact Hash Deduplication:** Run MD5 and perceptual hash (`imagehash.phash`) across all candidate images. Group duplicate bursts.
- **Split Preservation:**
  - TACO's sealed test split (374 images) remains strictly preserved.
  - General Waste Data's test split (1,042 images) remains independent.
  - Negative images are split 70% train / 15% val / 15% test.
- **Zero Cross-Split Contamination:** Near-duplicate clusters are assigned entirely to a single split.

---

## 7. Proposed Training Configuration & Success Criteria

### Training Specifications (Google Colab GPU)
- **Model:** YOLO11s (`yolo11s.pt`, AGPL-3.0)
- **Resolution:** $640 \times 640$ pixels
- **Batch Size:** 16 (automatic fallback to 8 if VRAM exceeds 14 GB)
- **Epochs:** 100 epochs (Early stopping patience = 20)
- **Optimizer:** AdamW ($\text{lr}_0 = 0.001$, $\text{decay} = 0.0005$)
- **Augmentation Tuning:**
  - `mosaic: 0.5` (reduced from 1.0 to prevent shrinking small litter below $16\times 16$ px)
  - `close_mosaic: 10` (disable mosaic for final 10 epochs for sharp localization)
  - `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4`, `fliplr: 0.5`
- **Estimated Compute Time:** ~2.5 to 3.5 hours on Google Colab NVIDIA T4 GPU.

### Success Criteria & Target Benchmarks
| Benchmark Metric | Current Baseline (v2) | Target Minimum for v3 Approval |
| :--- | :---: | :---: |
| **Overall mAP@50** | 30.7% | **$\ge 55.0\%$** |
| **Overall mAP@50–95** | 22.7% | **$\ge 38.0\%$** |
| **Precision** | 39.7% | **$\ge 60.0\%$** |
| **Recall** | 33.9% | **$\ge 50.0\%$** |
| **Glass mAP@50** | Low (<20%) | **$\ge 50.0\%$** |
| **Cardboard mAP@50** | Low (<25%) | **$\ge 50.0\%$** |
| **False Positive Rate on Negatives** | High (>70% false triggers) | **$\le 5.0\%$ false triggers** |
| **Specific Real-World Image Tests:** | | |
| • White phone charger | Fails (detected as paper) | **No detection (Clean rejection)** |
| • Table lamp | Fails (detected as cardboard) | **No detection (Clean rejection)** |
| • Clean shoes / slippers | Fails (detected as waste) | **No detection (Clean rejection)** |
| • Toothbrush on vanity | Fails (detected as plastic) | **No detection (Clean rejection)** |
| • Paper notebook | Fails (multiple cardboard) | **Correctly detected as paper / no false cardboard** |
| • Cardboard box | Detected with false classes | **Pure cardboard detection** |

---

## 8. Frontend & Backend Application Integration Plan

The improved model will be completely drop-in compatible with the existing EcoVision AI architecture:
1. **Model Weights:** The resulting checkpoint will be saved as `models/yolov11s_waste_v3.pt`.
2. **API Contract:** The 6 class IDs (`0: plastic`, `1: metal`, `2: cardboard`, `3: glass`, `4: paper`, `5: general_trash`) remain strictly identical. No database migrations or API schema changes are required.
3. **Disposal Guidance Engine:** The existing municipal segregation mapping (Blue recycling stream, separate glass stream, grey municipal trash stream) remains 100% valid.
4. **Confidence Calibration:** The backend prediction service will apply a calibrated confidence threshold ($\tau = 0.35$ for waste detection; below $\tau$ is treated as background / non-waste).

