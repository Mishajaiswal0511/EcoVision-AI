# EcoVision AI — Intelligent Waste Detection and Smart Segregation System
## Project Planning Report · Step 1 of N

> **Status:** Planning only. No code, no training, no dataset downloads have been executed yet.  
> **Workspace state at inspection:** Empty — safe to begin fresh.  
> **Date:** 2026-10-28

---

## Table of Contents

1. [Recommended Architecture and Project Folder Structure](#1-recommended-architecture-and-project-folder-structure)
2. [Dataset Comparison and Audit](#2-dataset-comparison-and-audit)
3. [Proposed Class Taxonomy and Data Requirements](#3-proposed-class-taxonomy-and-data-requirements)
4. [Baseline and Comparison Model Recommendations](#4-baseline-and-comparison-model-recommendations)
5. [Training and Evaluation Strategy](#5-training-and-evaluation-strategy)
6. [Novelty Features and Implementation Feasibility](#6-novelty-features-and-implementation-feasibility)
7. [Risks, Costs, License Restrictions, and Mitigations](#7-risks-costs-license-restrictions-and-mitigations)
8. [Staged Implementation Roadmap](#8-staged-implementation-roadmap)
9. [Open Decisions Requiring Verification](#9-open-decisions-requiring-verification)

---

## 1. Recommended Architecture and Project Folder Structure

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER (Browser)                      │
│          React + Vite + TailwindCSS (Frontend)          │
└───────────────────────┬────────────────────────────────-┘
                        │ HTTP (multipart upload / JSON)
                        ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                   │
│  ┌──────────────────┐   ┌──────────────────────────┐   │
│  │  REST Endpoints  │──▶│  Inference Service        │   │
│  │  (upload, history│   │  (YOLOv11 via Ultralytics)│   │
│  │   analytics)     │   │  + OOD Rejection Module   │   │
│  └──────────────────┘   └──────────────────────────┘   │
│  ┌──────────────────┐   ┌──────────────────────────┐   │
│  │  SQLite DB       │   │  XAI Module (EigenCAM /   │   │
│  │  (history store) │   │   GradCAM via hooks)      │   │
│  └──────────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        │
        (training happens separately in Google Colab)
                        │
┌─────────────────────────────────────────────────────────┐
│        Google Colab (GPU training environment)          │
│  YOLOv11 fine-tuning on curated waste dataset           │
│  Exports: best.pt → ONNX (optional)                     │
│  Logs: W&B / CSV metrics                                │
└─────────────────────────────────────────────────────────┘
```

**Why this split?**  
- Training is GPU-heavy and best done in Colab (free T4/A100).
- The local app only runs inference — it does not need a GPU if the model is small enough (YOLOv11n or YOLOv11s).
- FastAPI keeps Python's ecosystem available for CV operations; React/Vite provides a fast, modern UI.

---

### 1.2 Proposed Folder Structure

```
d:\Projects\ml_project\472\
├── PROJECT_PLAN.md              ← this file
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── detect.py        ← /api/detect endpoint
│   │   │   ├── history.py       ← /api/history endpoint
│   │   │   └── analytics.py     ← /api/analytics endpoint
│   │   ├── core/
│   │   │   ├── config.py        ← settings (paths, thresholds, no secrets)
│   │   │   └── model_loader.py  ← singleton model load on startup
│   │   ├── services/
│   │   │   ├── inference.py     ← YOLO inference + OOD rejection
│   │   │   ├── xai.py           ← Grad-CAM / EigenCAM wrapper
│   │   │   └── disposal.py      ← static disposal guidance lookup
│   │   ├── models/
│   │   │   └── db.py            ← SQLite schema (SQLModel or plain sqlite3)
│   │   └── main.py              ← FastAPI app + CORS + lifespan
│   ├── weights/                 ← trained .pt files (git-ignored)
│   ├── tests/
│   │   ├── test_inference.py
│   │   └── test_api.py
│   ├── requirements.txt
│   └── .env.example             ← template; never commit .env
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── DetectionCanvas.jsx
│   │   │   ├── DisposalCard.jsx
│   │   │   ├── HistoryPanel.jsx
│   │   │   └── AnalyticsDashboard.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Detect.jsx
│   │   │   ├── History.jsx
│   │   │   ├── Analytics.jsx
│   │   │   ├── Performance.jsx
│   │   │   └── About.jsx
│   │   ├── services/
│   │   │   └── api.js           ← Axios wrapper for backend calls
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css            ← Tailwind imports
│   ├── package.json
│   └── vite.config.js
│
├── training/                    ← Colab notebooks + training scripts
│   ├── notebooks/
│   │   ├── 01_data_audit.ipynb
│   │   ├── 02_dataset_prep.ipynb
│   │   ├── 03_train_yolov11.ipynb
│   │   ├── 04_train_yolov9.ipynb  ← comparison run
│   │   └── 05_evaluate.ipynb
│   ├── scripts/
│   │   ├── split_dataset.py
│   │   ├── label_audit.py
│   │   └── evaluate_ood.py
│   └── configs/
│       ├── yolov11_waste.yaml
│       └── yolov9_waste.yaml
│
├── data/
│   ├── raw/                     ← git-ignored; downloaded here
│   ├── processed/               ← standardised YOLO format
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/            ← kept sealed until final evaluation
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── ood_eval/                ← non-waste images for OOD testing
│   └── dataset.yaml             ← YOLO dataset config
│
├── evaluation/
│   ├── results/                 ← mAP CSVs, confusion matrices, PR curves
│   ├── real_world_test/         ← independently captured images
│   └── error_analysis/          ← FP/FN samples
│
├── docs/
│   ├── disposal_guide.json      ← static rules per class
│   └── methodology.md
│
├── .gitignore
└── README.md
```

> [!NOTE]
> `weights/` and `data/raw/` are in `.gitignore` — model files and raw datasets are not committed to the repository. Only configuration, code, and processed split manifests are versioned.

---

## 2. Dataset Comparison and Audit

### 2.1 Candidate Datasets

#### Dataset A — TACO (Trash Annotations in Context)
| Field | Details |
|---|---|
| **Full name** | TACO: Trash Annotations in Context |
| **Source URL** | http://tacodataset.org · https://github.com/pedropro/TACO |
| **Image count** | ~1,500 base images (community contributions may expand this) |
| **Annotation type** | COCO-format **instance segmentation** polygons; can be converted to bounding boxes |
| **Taxonomy** | 60 fine-grained categories under 28 super-categories (e.g., Bottle, Can, Carton, Cigarette, Cup, Plastic bag/wrapper, Paper, Lid) |
| **Pre-built splits** | No canonical fixed split; must be created manually or via the TACO API |
| **License** | CC BY 4.0 — free for research with attribution |
| **Scene type** | Real-world litter: streets, beaches, woods, parks |
| **Strengths** | Real-world conditions; diverse backgrounds and lighting; instance-level masks available |
| **Weaknesses** | Only ~1,500 images; severe class imbalance (thousands of cigarette butts, very few batteries); 60 classes → many classes have fewer than 50 examples; requires a Python download script with Flickr API |
| **Access method** | `pip install TACO` or clone GitHub repo + run `download.py` (requires Flickr API key) |
| **Overlaps with** | Roboflow "Waste Detection" datasets often re-use TACO images |

> [!WARNING]
> TACO images are sourced from Flickr and require a Flickr API key to download. Each image retains its original Flickr license (usually CC BY or CC BY-SA). **Verify each image's individual Flickr license before commercial use.**

---

#### Dataset B — Roboflow Universe "Waste Detection" (by Circus)
| Field | Details |
|---|---|
| **Full name** | Waste Detection Computer Vision Dataset |
| **Source URL** | https://universe.roboflow.com/circus/waste-detection-j6gab |
| **Image count** | ~4,659 base images; augmented exports can reach ~8,157 |
| **Annotation type** | Bounding boxes — exported directly in YOLOv8 / YOLOv11 format |
| **Classes** | Reported as multi-class waste including plastic, paper, metal, glass variants — **exact class list must be verified on the dataset page before use** |
| **Pre-built splits** | Yes: train/val/test via Roboflow export |
| **License** | CC BY 4.0 |
| **Scene type** | Mixed indoor/outdoor; real-world photos |
| **Strengths** | Already in YOLO format; direct API download available; good size; pre-split |
| **Weaknesses** | Class definitions may not match our target taxonomy exactly; augmented images may share originals with other Roboflow datasets; class distribution not independently verified |
| **Access method** | Roboflow API key (free tier) or manual browser download. `pip install roboflow` |
| **Overlaps with** | Potentially overlaps with TACO-derived Roboflow projects |

---

#### Dataset C — Garbage Classification 12-class (Kaggle, Mostafa Mohamed)
| Field | Details |
|---|---|
| **Full name** | Garbage Classification (12 classes) |
| **Source URL** | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| **Image count** | ~15,150 images |
| **Classes (12)** | paper, cardboard, biological (organic), metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, trash |
| **Annotation type** | **Classification only** — no bounding boxes; whole-image class labels |
| **Pre-built splits** | Not canonical; must create train/val/test manually |
| **License** | Open (Kaggle terms); check dataset card for specific terms |
| **Scene type** | Mostly isolated objects on clean backgrounds |
| **Strengths** | Largest image count; batteries class present; glass sub-types present |
| **Weaknesses** | **No bounding box annotations** — not usable directly for object detection; objects isolated from real backgrounds; "biological" and "trash" are vague; clothes and shoes don't fit our taxonomy |
| **Access method** | `kaggle datasets download mostafaabla/garbage-classification` (requires Kaggle API key) |
| **Overlaps with** | TrashNet (CardBoard, Glass, Metal, Paper, Plastic are common to both) |

---

#### Dataset D — TrashNet (Stanford)
| Field | Details |
|---|---|
| **Full name** | TrashNet |
| **Source URL** | https://github.com/garythung/trashnet |
| **Image count** | 2,527 images |
| **Classes (6)** | cardboard (403), glass (501), metal (410), paper (594), plastic (482), trash (137) |
| **Annotation type** | **Classification only** — no bounding boxes |
| **Pre-built splits** | No; images are in per-class folders |
| **License** | MIT License |
| **Scene type** | Objects photographed on white poster board under controlled lighting |
| **Strengths** | Clean, well-labeled; MIT license; small enough to process quickly |
| **Weaknesses** | **No bounding boxes**; white-background bias (models trained here fail on real-world images); very small; no organic, tissue, or e-waste |
| **Access method** | Download ZIP from GitHub releases (no API key required) |
| **Overlaps with** | Kaggle Garbage Classification dataset re-packages TrashNet data |

---

#### Dataset E — RealWaste (University of Wollongong, 2023)
| Field | Details |
|---|---|
| **Full name** | RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification |
| **Source URL** | https://archive.ics.uci.edu/dataset/908/realwaste · https://www.kaggle.com/datasets/joebeachcapital/realwaste |
| **Image count** | ~4,752 images |
| **Classes (9)** | Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation |
| **Annotation type** | **Classification only** — captured at a real landfill reception facility; whole-image labels |
| **Pre-built splits** | No canonical split |
| **License** | CC BY 4.0 (open access journal + UCI terms) |
| **Scene type** | Real landfill facility — cluttered, mixed backgrounds, authentic waste |
| **Strengths** | Most realistic scene content; includes organic/vegetation/textile; unique capture context |
| **Weaknesses** | **No bounding boxes**; landfill context may not generalize to household/urban scenes; Textile and Vegetation may not match our taxonomy |
| **Access method** | UCI ML Repository direct download or Kaggle API (no special key needed for UCI) |
| **Overlaps with** | Minimal overlap expected with TACO (different capture context) |

---

### 2.2 Compatibility Audit Summary

| Issue | Finding |
|---|---|
| **Bounding box availability** | Only TACO and the Roboflow Waste Detection dataset (B) have bounding box annotations. Datasets C, D, E are classification-only. |
| **Class name conflicts** | TACO uses "Plastic bag/wrapper", "Cup", "Lid"; Roboflow dataset uses broader terms. Merging requires explicit remapping. |
| **Duplicate risk** | Roboflow datasets frequently re-host TACO images. If both are used, hash-based deduplication is mandatory before splitting. |
| **Background bias** | TrashNet and Kaggle 12-class use clean/white backgrounds. Mixing these with TACO real-world images without careful weighting will harm generalization. |
| **Class count mismatch** | It is impossible to directly merge a 6-class, a 12-class, and a 60-class dataset without a well-defined mapping table. This must be done before any training. |
| **Organic/tissue gap** | No dataset provides strong bounding-box-annotated examples of food waste in realistic household settings. This is a known gap. |

### 2.3 Recommended Dataset Strategy

**Primary detection dataset:** Roboflow Universe "Waste Detection" (Dataset B)
- Already in YOLO bounding-box format
- Free download with Roboflow API key (free account)
- Good class variety for a student project
- Must verify exact class list and distribution on the dataset page before confirming

**Optional enrichment (if classes are compatible):** TACO (Dataset A)
- Adds real-world litter images with instance segmentation
- Must be converted from COCO segmentation to YOLO bounding boxes using a script
- **Do not combine with Dataset B before running a hash-based duplicate check**
- Requires Flickr API key for the downloader

**Classification datasets (C, D, E) — Do NOT use directly for detection**  
They may be used later as a supplementary pre-training or data-augmentation source only after bounding box annotations are manually added or generated via a labeling tool (e.g., Label Studio, Roboflow Annotate).

> [!IMPORTANT]
> Download instructions for Step 2: Dataset B requires creating a free Roboflow account, generating a private API key, and using `pip install roboflow`. The key must be stored in a `.env` file and never committed. TACO requires a Flickr API key from https://www.flickr.com/services/apps/create/. Neither download is automated in this plan — they require manual account creation.

---

## 3. Proposed Class Taxonomy and Data Requirements

### 3.1 The Core Problem: Material vs. Object

A critical design decision: should we detect **materials** (Plastic, Glass, Metal) or **objects** (Bottle, Can, Newspaper)?

- **Material taxonomy** is aligned with how recycling bins are labeled in real life (users need to know *where to put it*, not just *what it is*).
- **Object taxonomy** is what datasets like TACO actually annotate (specific objects, not materials).
- **Recommended approach:** Use a two-level system — detect objects, map them to material categories for disposal advice. The model trains on visible object classes; the application presents material-level guidance.

### 3.2 Proposed Detection Classes (What the Model Sees)

The following 8 classes are proposed as the **training label set**. Final class selection depends on actual available annotated examples in the chosen datasets — this must be verified in Step 2.

| Class ID | Detection Label | Material Category | Disposal Method |
|---|---|---|---|
| 0 | `plastic_bottle` / `plastic` | Plastic | Dry recycling bin |
| 1 | `paper` | Paper | Dry recycling bin |
| 2 | `cardboard` | Cardboard | Dry recycling bin |
| 3 | `glass` | Glass | Glass recycling / separate glass bin |
| 4 | `metal` / `can` | Metal | Dry recycling bin |
| 5 | `organic` / `food_waste` | Organic | Wet / compost bin |
| 6 | `battery` | Hazardous | Specialist e-waste drop-off — **Safety warning** |
| 7 | `general_trash` | General | General waste bin |

**Classes NOT included initially (and why):**
- **Tissue / wet wipes:** Very few annotated examples in available datasets; texture similar to paper creates confusion.
- **E-waste (circuit boards, chargers):** Virtually no publicly available bounding-box training data in sufficient quantity. Battery class is retained as a proxy.
- **Clothes / Textile:** RealWaste has this as image-level label only; insufficient detection examples.
- **Shoes:** Not useful for typical waste detection scenarios.

> [!CAUTION]
> Do not add a class to training unless there are **at least 300–500 annotated bounding box instances** across diverse images. Fewer instances lead to high false-positive rates and confuse the model during evaluation.

### 3.3 Handling Non-Waste Objects (Open-Set Rejection)

Standard YOLO models operate under a **closed-world assumption**: the model has never been trained to say "I don't know." If you hold up a laptop or a bed pillow, the model will still output the closest matching class with a confident score — this is by design, not a bug.

**Why this matters for EcoVision AI:**  
A user might photograph a kitchen counter with a coffee mug, a knife, and a plastic bottle. The mug and knife are not waste — but a naive model might label the mug as "glass" with 80% confidence.

**Proposed testable OOD-rejection strategy (three-layer approach):**

1. **Confidence threshold filter (Layer 1 — simple, always present):**
   - Set a tuned minimum confidence threshold (start at 0.40; tune on a validation OOD set).
   - Any detection below threshold → display "Unknown / Not a supported waste item."
   - Known limitation: deep networks are often overconfident even on OOD inputs; this layer alone is insufficient.

2. **OOD evaluation set (Layer 2 — measurement and tuning):**
   - Collect ~200 images of common non-waste household objects (laptops, chairs, clothes hangers, books, cups, utensils).
   - Run the trained model on these and record how many are confidently classified as waste.
   - Use this to set a threshold that gives an acceptable false-acceptance rate.
   - This evaluation set must never be used for training or for tuning the detector weights.

3. **Future consideration — Open-vocabulary or auxiliary discriminator (Layer 3 — research extension):**
   - YOLO-World (an open-vocabulary extension of YOLO) can accept text prompts and compute semantic similarity. It has better-calibrated uncertainty on OOD inputs.
   - An auxiliary binary "is this a waste object?" classifier trained on waste vs. non-waste images can be added as a pre-filter.
   - These are non-trivial to implement and are flagged as optional novelty features, not baseline requirements.

---

## 4. Baseline and Comparison Model Recommendations

### 4.1 Model Comparison

| Model | Variant | Parameters | mAP@50 (COCO) | Inference (T4) | Colab compat. | License | Notes |
|---|---|---|---|---|---|---|---|
| **YOLOv8** | YOLOv8s | ~11M | ~44.9 | ~18 ms | ✅ | AGPL-3.0 | Extremely mature; vast documentation; strong community; anchor-free |
| **YOLOv11** | YOLOv11s | ~9.4M | ~47.0 | ~13–14 ms | ✅ | AGPL-3.0 | Fewer parameters than v8 for similar/better accuracy; latest Ultralytics release |
| **YOLOv9** | YOLOv9s | ~7.1M (GELAN) | ~46.8 | ~20 ms | ✅ | GPL-3.0 | PGI architecture excels on small/occluded objects; slower than v8/v11 |
| **RT-DETR** | RT-DETRv2-S | ~20M | ~48.1 | ~25 ms | ⚠️ (more RAM) | Apache 2.0 | Transformer-based; better global context; heavier; steeper learning curve |

> [!NOTE]
> All mAP values above are on the standard COCO val2017 benchmark. Performance on a custom waste dataset will differ substantially — the above are for relative comparison only. Do not quote these as your project results.

**License note — AGPL-3.0:** YOLOv8 and YOLOv11 use the AGPL-3.0 license. This is a **copyleft** license — if the application is deployed as a public network service, the source code must be made available. For a student project or private deployment, this is not a problem. For a commercial SaaS product, a commercial Ultralytics license would be required.

### 4.2 Recommended Models

**Baseline model:** `YOLOv11s` (small variant)
- Reason: Best balance of accuracy, speed, and parameter efficiency among current Ultralytics models.
- Fewer parameters than YOLOv8 for comparable or better mAP.
- Excellent documentation; Ultralytics Python API works identically to YOLOv8 (minimal code changes).
- Colab T4 GPU trains comfortably within free-tier time limits.
- Inference on a CPU (local Windows) is feasible for static image detection.

**Comparison model:** `YOLOv9s` (compact/small GELAN variant)
- Reason: Architecturally different (PGI + GELAN design); provides a meaningful scientific comparison.
- Expected to perform better on small or overlapping waste objects (e.g., bottle caps, crumpled paper).
- Slower inference — useful to quantify the accuracy vs. speed trade-off.
- Provides viva-ready discussion: "We compared two architecturally distinct detectors to understand which design better handles our real-world waste scenarios."

**Why NOT the nano variant as baseline:**  
YOLOv11n is very fast but sacrifices accuracy noticeably. For a portfolio project where detection quality matters, the small variant is the correct starting point. Nano can be tested as a third lightweight option.

**Why NOT RT-DETR as baseline:**  
RT-DETR is a transformer detector with a steeper implementation curve and higher memory requirement. It is not necessary for the baseline and can be added later as an advanced comparison if time permits.

---

## 5. Training and Evaluation Strategy

### 5.1 Data Split Policy

```
Total annotated images (after deduplication and label mapping)
        │
        ├──── Train: 70%  ──→ used for gradient updates only
        ├──── Val:   15%  ──→ used for hyperparameter tuning and early stopping
        └──── Test:  15%  ──→ SEALED until final evaluation; never used to tune the model
```

**Near-duplicate prevention:**
- Compute perceptual hashes (pHash) for every image before splitting.
- Images with pHash distance < threshold (typically Hamming distance < 10) are considered near-duplicates.
- Only one copy is kept; the rest are removed before splitting.
- This prevents a "leaked" evaluation where the model has seen essentially the same image during training.

**Cross-dataset deduplication:**  
If TACO images are added to the Roboflow dataset, run the same pHash check across the combined pool to catch re-hosted images.

### 5.2 Why Detection mAP ≠ Classification Accuracy

This is a critical concept for your viva:

- **Classification accuracy** answers: "Given exactly one object, what class is it?" — a single number, no localization.
- **Object detection mAP** measures two things simultaneously: (a) did the model find the object (localization with IoU ≥ 0.50), and (b) did it assign the correct class?
- A model can have **high classification accuracy but low mAP** if it finds objects in the wrong location, finds duplicates, or misses objects entirely.
- **mAP@50** is more lenient (bounding box must overlap ground truth by ≥ 50%).
- **mAP@50–95** averages over IoU thresholds from 0.50 to 0.95 in 0.05 steps — a much stricter metric that penalizes imprecise box placement.
- **High validation mAP does not guarantee real-world performance** because: (a) the validation set shares the same data distribution as training; (b) real-world images differ in lighting, angle, occlusion, and background; (c) near-duplicates inflate apparent performance.

### 5.3 Evaluation Metrics to Report

| Metric | What it measures | How to compute |
|---|---|---|
| mAP@50 | Detection at 50% IoU overlap | `model.val()` in Ultralytics |
| mAP@50–95 | Detection across strict IoU range | `model.val()` in Ultralytics |
| Per-class Precision | Ratio of correct detections | Ultralytics results CSV |
| Per-class Recall | Ratio of detected real objects | Ultralytics results CSV |
| Confusion matrix | Which classes are confused | `model.val(plots=True)` |
| F1 score | Harmonic mean of P and R | Computed from P and R |
| FP rate on OOD set | How often model wrongly detects non-waste as waste | Custom script on OOD eval set |

### 5.4 Evaluation Protocol

**Phase 1 — Validation set evaluation (during training):**
- Run after every epoch on the validation set.
- Use for early stopping and learning rate scheduling.
- Do NOT use to decide which model architecture "wins."

**Phase 2 — Test set evaluation (after training is complete):**
- Run exactly once per model variant.
- Report all metrics from this run.
- Never go back and re-train specifically to improve test set numbers (this is test set leakage).

**Phase 3 — Real-world evaluation (after test set evaluation):**
- Collect 50–100 independently photographed images of waste and non-waste (not from any training dataset).
- Cover: different lighting, outdoor, indoor, multiple objects per image, partially occluded objects.
- Run inference and manually review: count true positives, false positives, missed detections.
- Report qualitative findings alongside the quantitative test metrics.

**Phase 4 — OOD evaluation:**
- Run the trained model on the ~200 non-waste household object images.
- Record how many are incorrectly detected as waste categories and with what confidence.
- Use results to justify the chosen confidence threshold.

---

## 6. Novelty Features and Implementation Feasibility

### 6.1 Feature 1 — Confidence-Aware Predictions and Unknown-Object Rejection

**What it does:** Instead of always showing a label, the UI displays:
- High confidence (≥ 0.70): Label + category + disposal guidance with green indicator.
- Medium confidence (0.40–0.70): Label shown with amber indicator and a "low confidence" warning.
- Below threshold (< 0.40): "Unknown / Not a supported waste object" with grey indicator.

**Why it's meaningful:** Standard YOLO demos always show labels regardless of confidence, which misleads users when the model is uncertain. This feature makes uncertainty a first-class UI element.

**Implementation:** Confidence thresholds in `inference.py`; conditional rendering in `DetectionCanvas.jsx`. **Feasible in Week 5–6.**

---

### 6.2 Feature 2 — Explainable Predictions (Gradient-Based Heatmaps)

**What it does:** For each detection, an optional "Why?" button generates a Grad-CAM or EigenCAM heatmap overlaid on the detected region, highlighting which pixels activated the model.

**Important caveat to communicate clearly:** A Grad-CAM heatmap shows *which pixels influenced the model's decision*, not *why the object is waste* or *that the model understood the object*. This is activation visualization, not a proof of reasoning.

**Implementation:** `pip install YOLOv8-Explainer` or PyTorch hooks on the trained model. The heatmap is generated on demand (not for every detection, since it adds ~500 ms). The backend `/api/explain` endpoint returns a base64-encoded heatmap image. **Feasible in Week 7. Moderate complexity.**

---

### 6.3 Feature 3 — Real-World Robustness Evaluation Suite

**What it does:** The "Model Performance" page in the UI displays:
- Genuine mAP@50, mAP@50–95, per-class P/R from the sealed test set.
- A confusion matrix heatmap showing which classes are most often confused.
- Sample false positive and false negative images from the error analysis set.

**Why it's meaningful:** Many student projects show only headline accuracy numbers. Showing actual failure cases and confusion patterns demonstrates scientific maturity.

**Implementation:** Generate all metrics during training in Colab. Save as JSON. Load JSON in the Performance page component. **Feasible in Week 6–7. Low complexity.**

---

### 6.4 Feature 4 — Smart Disposal Recommendations

**What it does:** Each detected class maps to structured disposal guidance:
- Which bin to use (color-coded for India: blue = dry recyclables, green = wet/organic).
- Whether the item needs cleaning before disposal.
- Special handling instructions (e.g., "Remove the cap before recycling the bottle").
- A safety warning for batteries and e-waste (red banner).

**Implementation:** Static lookup table in `disposal.py` / `disposal_guide.json`. Displayed in `DisposalCard.jsx`. **Feasible in Week 5. Low complexity.**

---

### 6.5 Feature 5 — Error Analysis Dashboard

**What it does:** A dedicated dashboard page that loads the saved error analysis JSON and displays:
- Bar chart: Most commonly confused class pairs.
- Gallery: Sample FP and FN images with predicted vs. ground truth labels.
- Slider: Filter by confidence to explore threshold sensitivity.

**Why it's meaningful:** Demonstrates that the developer understands model limitations and has investigated failure modes — a highly impressive quality for a viva or portfolio.

**Implementation:** Pre-computed during evaluation in Colab. Loaded as JSON in the frontend. Visualization with Recharts (React charting library). **Feasible in Week 7–8. Moderate complexity.**

---

### 6.6 Features NOT included and Why

| Feature | Reason excluded |
|---|---|
| Live video detection stream (real-time camera) | Requires WebSocket or WebRTC streaming; significantly more complex; YOLO inference latency on CPU makes real-time video impractical without GPU. Marked as a stretch goal for Step 5. |
| Custom trained segmentation masks | Insufficient segmentation-annotated data; bounding boxes are the reliable focus. |
| Mobile app | Out of scope for Phase 1. |
| Claiming "95% accuracy" | Not a real result; accuracy cannot be quoted until evaluation is run on the actual trained model. |

---

## 7. Risks, Costs, License Restrictions, and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Insufficient annotated bounding-box images | High | Audit Dataset B before committing; plan for Roboflow Annotate if needed |
| TACO download fails (Flickr API limits) | Medium | Use Roboflow-hosted TACO derivatives as fallback |
| Class imbalance (e.g., too few battery images) | High | Apply class-weighted loss in YOLO config; use targeted augmentation |
| Model memorizes white-background images | High | Exclude or down-weight TrashNet/clean-background images; prefer real-world data |
| Test set leakage (accidentally tuning on test data) | Critical | Seal test set folder; only unpack it once, for the final evaluation |
| Google Colab session timeout during training | Medium | Use Colab Pro or save checkpoints every 5 epochs to Google Drive |
| AGPL-3.0 licensing (YOLOv8/v11) | Low for student use | Fine for academic/portfolio projects; note restriction in docs |
| Duplicate images across datasets | High | Run pHash deduplication before any split |
| Non-waste objects falsely classified as waste | Medium | OOD eval set + confidence threshold; document limitations clearly |
| Real-world performance much lower than validation | High | Do NOT skip real-world evaluation phase; document gap honestly |

### 7.1 Cost Summary

| Resource | Cost |
|---|---|
| Google Colab (free tier, T4 GPU) | Free; limited to ~4 hrs/session; sufficient for YOLOv11s training |
| Google Colab Pro | ~\$10/month; recommended for longer training runs |
| Roboflow free account | Free; includes dataset hosting and API access |
| Kaggle API | Free |
| Flickr API key (TACO) | Free |
| All mentioned Python libraries | Free and open-source |
| Deployment (optional, Render / Railway / Hugging Face Spaces) | Free tier available |

---

## 8. Staged Implementation Roadmap

```
Step 1  ─── PLANNING (current step) ─────────────────── Week 1
        ✅ Architecture defined
        ✅ Datasets researched and audited
        ✅ Class taxonomy proposed
        ✅ Model selection recommended
        ✅ Evaluation plan defined

Step 2  ─── DATA ACQUISITION AND PREPARATION ──────────── Week 2
        [ ] Create Roboflow account and download Dataset B
        [ ] Audit actual class list and distribution
        [ ] Run pHash deduplication
        [ ] Define label mapping table
        [ ] Create 70/15/15 split with stratification
        [ ] Generate dataset.yaml for YOLO
        [ ] Seal test split folder
        [ ] Optionally: download and convert TACO subset

Step 3  ─── BASELINE MODEL TRAINING ──────────────────── Week 3
        [ ] Set up Colab training notebook
        [ ] Train YOLOv11s on curated dataset
        [ ] Monitor training curves (loss, mAP)
        [ ] Save best checkpoint to Google Drive
        [ ] Run validation evaluation; record results

Step 4  ─── COMPARISON MODEL AND EVALUATION ───────────── Week 4
        [ ] Train YOLOv9s on identical dataset and config
        [ ] Evaluate both models on val set
        [ ] Run final evaluation on SEALED test set (once per model)
        [ ] Generate confusion matrices, PR curves, per-class results
        [ ] Collect and run OOD evaluation
        [ ] Collect 50–100 real-world independent test images
        [ ] Document all genuine results

Step 5  ─── BACKEND DEVELOPMENT ──────────────────────── Week 5
        [ ] Set up FastAPI project structure
        [ ] Implement model loading singleton
        [ ] Implement /api/detect endpoint with confidence tiers
        [ ] Implement disposal guide lookup
        [ ] Implement history store (SQLite)
        [ ] Write pytest tests for endpoints
        [ ] Load best model weights

Step 6  ─── FRONTEND DEVELOPMENT ─────────────────────── Week 6
        [ ] Set up React + Vite + TailwindCSS
        [ ] Build UploadZone and DetectionCanvas components
        [ ] Display bounding boxes + confidence indicators
        [ ] Display DisposalCard per detected class
        [ ] Build HistoryPanel
        [ ] Build Performance page with genuine metrics
        [ ] Build About and Methodology pages

Step 7  ─── NOVELTY FEATURES ─────────────────────────── Week 7
        [ ] Implement EigenCAM / GradCAM XAI module
        [ ] Implement /api/explain endpoint
        [ ] Build Error Analysis Dashboard (loaded from pre-computed JSON)
        [ ] Refine OOD rejection confidence threshold based on OOD eval

Step 8  ─── POLISH, TESTING, AND DOCUMENTATION ─────── Week 8
        [ ] End-to-end integration testing
        [ ] Responsive design audit
        [ ] README and methodology documentation
        [ ] Limitations section in UI
        [ ] Optional: export model to ONNX for faster CPU inference
        [ ] Optional: deploy to Hugging Face Spaces or Render

```

---

## 9. Open Decisions Requiring Verification

The following decisions cannot be finalized from research alone. They must be verified by actually accessing the datasets and tools in Step 2.

| Decision | What needs verification | How to verify |
|---|---|---|
| Exact class list in Dataset B | The Roboflow "Waste Detection" dataset page must be opened to confirm exact label names and instance counts | Open the dataset page; download and inspect a sample |
| Viable image count per class | Minimum viable class requires ~300–500 instances; total dataset size must be confirmed | Download → count annotations per class with a script |
| TACO download feasibility | TACO requires Flickr API key and a working download script; this must be tested | Create Flickr developer account; run `download.py` on a small batch |
| Duplicate rate between datasets | Unknown without running pHash comparison | Combine raw image sets → run pHash script |
| OOD threshold calibration | Confidence threshold for unknown rejection cannot be set without first training the model | Set after Step 4 evaluation |
| Colab training time for YOLOv11s | Depends on final dataset size; estimate 1–3 hours for ~5,000 images at 100 epochs on T4 | Run a 5-epoch smoke test before committing to full run |
| Battery class viability | Number of battery annotations in Dataset B is unknown | Check annotation counts in Step 2 |
| Tissue/wet-wipe class viability | No confirmed bounding-box source found | Search Roboflow Universe specifically for "tissue waste" before Step 2 |
| YOLOv9 AGPL vs. GPL distinction | YOLOv9's original paper repo uses GPL-3.0; verify which repo/version is used | Check the exact GitHub repo license file |
| Hardware for local inference | Whether YOLOv11s CPU inference on the user's Windows machine is fast enough for the web app | Run a quick benchmark on the machine after exporting weights |

---

## Appendix: Useful Commands (Reference Only — Not to Be Run Yet)

```bash
# Step 2: Install libraries (do not run until workspace setup)
pip install ultralytics roboflow pillow imagehash pytest

# Roboflow dataset download (API key from .env, never hardcoded)
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("workspace-name").project("waste-detection-j6gab")
dataset = project.version(1).download("yolov11")

# Kaggle dataset download
kaggle datasets download mostafaabla/garbage-classification

# pHash deduplication (concept, not final code)
import imagehash
from PIL import Image
hash1 = imagehash.phash(Image.open("image1.jpg"))
hash2 = imagehash.phash(Image.open("image2.jpg"))
if abs(hash1 - hash2) < 10:
    print("Near-duplicate found")
```

> [!IMPORTANT]
> The commands above are for reference planning only. No download or training has been performed. The project workspace is empty. Step 2 begins only after this plan is reviewed and approved.

---

*Report prepared: 2026-10-28 | EcoVision AI Project | Step 1 of 8*

