# EcoVision AI — YOLO11s Baseline Audit Report

**Date:** October 29, 2026  
**Auditor:** Senior ML Engineer & CV Architect  
**Target:** YOLO11s Baseline Training & Dataset Pipeline  
**Workspace:** `D:\Projects\ml_project\472`  
**Dataset:** Processed 6-Class TACO Dataset (`data/processed`)  

---

## 1. Executive Summary & Root Cause Verdict

An exhaustive, non-destructive audit of the dataset preparation scripts, raw source annotations, processed label files, and training configuration was conducted to investigate the baseline validation metrics:
- **Validation Metrics Reported:** 379 images, 1,076 instances, **Precision: 0.463**, **Recall: 0.144**, **mAP50: 0.119**, **mAP50-95: 0.0538**.
- **Observed Symptoms:** Cardboard, glass, and paper exhibited near-zero detection performance; prediction plots showed massive overlapping bounding boxes covering wide canvas areas.

### The Smoking Gun (Confirmed Root Cause)
The poor baseline performance is **not** an architectural failure of YOLO11s or model underfitting. It is caused by a **fatal coordinate parsing bug** in `scripts/label_mapper.py`:
- The raw TACO dataset exported from Roboflow (`data/raw/garbage_taco-official-all-class_v6`) is in **YOLOv11 Instance Segmentation format (polygons)**, where each line consists of `class_id x1 y1 x2 y2 x3 y3 ... xn yn` (ranging from 9 to 817 values per line).
- `scripts/label_mapper.py` (lines 122–130) treated the first four coordinate floats (`parts[1:5]`) as bounding box parameters `cx, cy, w, h`.
- Consequently:
  $$\text{center } (cx, cy) = (x_1, y_1) \quad (\text{vertex 1 of polygon})$$
  $$\text{dimensions } (w, h) = (x_2, y_2) \quad (\text{absolute position of vertex 2!})$$
- Every bounding box in `data/processed` was assigned a width and height equal to the canvas coordinates of the second polygon vertex. This inflated median object area by **65.4x** (from $0.003557$ true area to $0.232685$ bogus area), turning tiny litter into giant canvas-covering boxes and shifting box centers to polygon edges.

---

## 2. Incontrovertible Evidence & Empirical Findings

### Finding 1: Polygon vs BBox Format Discrepancy
- **Raw TACO Annotation Audit:**
  - Total annotation lines scanned: **8,152**
  - Lines with exactly 5 tokens (standard bounding box): **0 (0.0%)**
  - Lines with $>5$ tokens (polygon segmentation vertices): **8,152 (100.0%)**
  - Minimum tokens per line: **9** (4 polygon vertices + class ID)
  - Maximum tokens per line: **817** (408 polygon vertices + class ID)

- **Code Flaw in `scripts/label_mapper.py` (Lines 123–130):**
  ```python
  parts = line.split()
  if len(parts) < 5:
      continue
  # FLAW: blindly treats polygon vertices (x1, y1) and (x2, y2) as (cx, cy, w, h)
  src_id = int(parts[0])
  cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
  new_line = f"{target_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
  ```

- **Direct Numerical Comparison on Raw TACO Samples:**
  | File & Class | Polygon Vertices | True BBox `(cx, cy, w, h)` | Bogus Processed BBox `(cx, cy, w, h)` | Distortion Factor |
  | :--- | :---: | :---: | :---: | :---: |
  | `000000_JPG` (`Other plastic wrapper`) | 42 | `(0.6002, 0.6330, 0.0268, 0.0670)` | `(0.6005, 0.6647, 0.6030, 0.6665)` | Width: **22.5x**, Height: **10.0x** |
  | `000000_JPG` (`Disposable plastic cup`) | 23 | `(0.4257, 0.5470, 0.0915, 0.0542)` | `(0.4216, 0.5616, 0.4310, 0.5650)` | Width: **4.7x**, Height: **10.4x** |
  | `000001_jpg` (`Squeezable tube`) | 13 | `(0.4636, 0.3531, 0.0670, 0.1118)` | `(0.4800, 0.3900, 0.4828, 0.3824)` | Width: **7.2x**, Height: **3.4x** |
  | `000001_jpg` (`Paper cup`) | 69 | `(0.4737, 0.5252, 0.1864, 0.1386)` | `(0.3936, 0.5753, 0.3996, 0.5800)` | Width: **2.1x**, Height: **4.2x** |

### Finding 2: Why Automated Checks Failed to Detect the Bug
In `scripts/check_annotations.py`:
1. **Line 126:** Checked `if len(parts) < 5: continue`. Because polygon lines contain between 9 and 817 tokens, they easily passed this check.
2. **Lines 151–159:** Checked `0.0 <= cx <= 1.0` and `0.0 <= w <= 1.0`. Because all polygon vertices in Roboflow exports are normalized between 0.0 and 1.0, $(x_1, y_1)$ and $(x_2, y_2)$ were always within $[0, 1]$.
3. **Missing Calculation:** The script never verified bounding box boundaries $x_2 = cx + w/2$ or $y_2 = cy + h/2$. When computed properly, **3,093 out of 8,152 boxes (37.9%)** extend outside the image canvas.

### Finding 3: Object Size Distribution Distortion
A comparison of the bounding box size distribution in the current processed dataset versus the mathematically derived true bounding boxes:

| Metric | Bogus Processed Labels | True Bounding Boxes (from Polygons) |
| :--- | :---: | :---: |
| **Small Objects (<32×32 px / <0.25% canvas)** | **44 (0.5%)** | **3,672 (45.0%)** |
| **Medium Objects (32–96 px / 0.25%–2.25%)** | **257 (3.2%)** | **2,712 (33.3%)** |
| **Large Objects (>96×96 px / >2.25% canvas)** | **7,818 (96.3%)** | **1,768 (21.7%)** |
| **Minimum Area** | 0.000001 | 0.000010 |
| **Median Area** | **0.232685** | **0.003557** (65.4x smaller) |
| **75th Percentile Area** | **0.468200** | **0.017892** (26.2x smaller) |

**Impact on YOLO11s Training:**  
The model was heavily penalized for predicting true-sized waste objects. Instead, gradient updates forced anchor-free heads to predict massive boxes spanning nearly a quarter to half of the entire 640×640 image canvas. During validation, predictions naturally produced overlapping, screen-spanning bounding boxes with low IoU overlap against target metrics.

---

## 3. Dataset Pairing, Label Validation & Overlap Audit

### Label Verification Across Splits
- **Image-Label Pairing:** 100% paired. Exactly 1,790 image/label pairs in `train`, 379 in `val`, and 374 in `test`. No orphaned files.
- **Empty / Corrupt Files:** 0 empty files across all splits.
- **Class IDs:** Strictly contiguous $\{0, 1, 2, 3, 4, 5\}$ matching `data/dataset.yaml`.

### Duplicate and Overlapping Annotations Audit
Due to polygon-to-box distortion and native TACO multi-object scene annotations:
- **Exact / Near Duplicates ($\text{IoU} > 0.90$):**
  - `train`: **323 duplicate pairs** (244 same class, **79 different class conflicts**).
  - `val`: **30 duplicate pairs** (24 same class, **6 different class conflicts**).
- **High-IoU Overlaps ($0.60 < \text{IoU} \le 0.90$):**
  - `train`: **3,246 overlapping pairs** (2,189 same class, **1,057 conflicting class overlaps**).
  - `val`: **315 overlapping pairs** (228 same class, **87 conflicting class overlaps**).
- **Sample Ground Truth Contradictions in Raw TACO:**
  - `000023_JPG.rf.870bd41d2ba159c35b0903d367db748a`: Box at `(0.450, 0.658)` is labeled simultaneously as `Other carton` (mapped to `cardboard`) and `Cigarette` (mapped to `general_trash`) with $\text{IoU} = 0.959$.
  - `000002_JPG.rf.610e0ec78d4d5adc82178d691606dc70`: Overlapping boxes labeled as `Plastic film` (plastic) and `Normal paper` (paper) with $\text{IoU} = 0.902$.

---

## 4. Class Mapping & Taxonomy Audit

Audit of all 59 raw TACO classes against the 6-class target taxonomy in `data/label_mapping.json`:

| Target Class | Raw Class Count | Raw Instances | Primary Constituents | Visual & Material Ambiguities |
| :--- | :---: | :---: | :--- | :--- |
| **`0: plastic`** | 22 | 3,875 | `Plastic film` (789), `Clear plastic bottle` (484), `Other plastic` (456), `Other plastic wrapper` (446), `Plastic bottle cap` (348). | `Crisp packet` (66) is metallized film; `Plastic straw` (280) vs `Paper straw` (7) are visually similar; `Disposable plastic cup` (176) vs `Paper cup` (117) vs `Foam cup` (24). |
| **`1: metal`** | 8 | 900 | `Drink can` (370), `Pop tab` (161), `Metal bottle cap` (133), `Aluminium foil` (106), `Food Can` (55). | `Aluminium foil` is crumpled/shiny, often visually identical to `Crisp packet` interior; `Metal bottle cap` vs `Plastic bottle cap`. |
| **`2: cardboard`** | 7 | 437 | `Other carton` (159), `Corrugated carton` (114), `Drink carton` (73), `Meal carton` (57), `Egg carton` (21). | `Drink carton` (Tetra Pak) is a composite material (paperboard + polyethylene + foil); visually distinct from corrugated shipping boxes. |
| **`3: glass`** | 4 | 396 | `Broken glass` (193), `Glass bottle` (181), `Glass cup` (11), `Glass jar` (11). | `Broken glass` consists of tiny, sharp transparent shards; regular glass bottles are whole/cylindrical. Severe imbalance (only 396 instances total). |
| **`4: paper`** | 7 | 417 | `Normal paper` (136), `Paper cup` (117), `Tissues` (75), `Paper bag` (51), `Wrapping paper` (17). | `Paper cup` is polyethylene-lined; crumpled `Tissues` are visually ambiguous against crumpled `Plastic film` or white bags. |
| **`5: general_trash`** | 4 | 2,094 | `Cigarette` (1,128), `Unlabeled litter` (903), `Rope - strings` (51), `Aluminium blister pack` (12). | **Severe intra-class variance:** 53.9% of class consists of tiny cigarette butts; 43.1% consists of arbitrary unlabeled debris. `Aluminium blister pack` (composite) is mapped here while `Aluminium foil` is `metal`. |
| **`Excluded (null)`** | 4 | 30 | `Food waste` (13), `Shoe` (11), `Battery` (4), `Carded blister pack` (2). | Correctly excluded due to negligible sample counts and non-segregation categories. |

---

## 5. Training Configuration & Checkpoint Analysis

- **Configuration:** `training/configs/yolov11_waste.yaml`
  - `imgsz: 640`: Standard resolution, but when $45.0\%$ of objects are $<32\times 32$ px, downscaling occurs.
  - `mosaic: 1.0`: Stitches 4 images into one, reducing small objects by another $2\times$ (down to $<16\times 16$ px). With corrupt box dimensions, mosaic created catastrophic spatial chaos.
  - `batch: 16`, `optimizer: AdamW`, `lr0: 0.001`: Standard transfer learning setup.
- **Class Ordering:** Strictly identical between `data/dataset.yaml` and `data/label_mapping.json`.
- **Checkpoint Validation:** The user's Colab validation results (379 images, 1,076 instances) exactly match the processed `val` split annotation count. The checkpoint was validated on the intended data split, confirming that the low recall (0.144) was driven entirely by training on distorted pseudo-boxes.

---

## 6. Categorization of Issues

### A. Confirmed Bugs (Definitive Code Proof)
1. **Polygon Vertex Truncation (`scripts/label_mapper.py`):** Polygons with $N$ vertices were parsed as $(cx, cy, w, h) = (x_1, y_1, x_2, y_2)$.
2. **Permissive Annotation Checker (`scripts/check_annotations.py`):** Failed to reject lines with $>5$ tokens and failed to test boundary limits $cx \pm w/2, cy \pm h/2$.
3. **Contradictory Ground Truth Annotations:** 79 duplicate pairs in `train` and 6 in `val` where the exact same bounding box is annotated as two different classes.

### B. Contributing Factors (Dataset & Training Dynamics)
1. **Severe Class Imbalance:** Plastic (3,875) and General Trash (2,094) comprise $73.5\%$ of annotations. Cardboard (437), Glass (396), and Paper (417) have nearly $10\times$ fewer examples.
2. **Small-Object Dominance:** $45.0\%$ of real objects are $<32\times 32$ pixels (Cigarettes, bottle caps, tabs), which require specialized detection heads or higher input resolution.
3. **High Intra-Class Variance in `general_trash`:** Combining tiny cigarette butts with arbitrary unidentified litter makes feature learning difficult.

### C. What Remains Uncertain
- Exact impact of offline Roboflow augmentations (horizontal/vertical flips, 90-degree rotations) on orientation-sensitive waste items.
- Optimal confidence threshold for NMS on the corrected dataset.

---

## 7. Single Highest-Priority Next Action

**Update `scripts/label_mapper.py` to correctly compute bounding boxes from polygon segmentation masks.**

For any line where $\text{len}(\text{parts}) > 5$:
$$\text{Extract coordinates: } xs = [x_1, x_2, \dots, x_n], \quad ys = [y_1, y_2, \dots, y_n]$$
$$x_{\min} = \min(xs), \quad x_{\max} = \max(xs), \quad y_{\min} = \min(ys), \quad y_{\max} = \max(ys)$$
$$w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$
$$cx = x_{\min} + \frac{w}{2}, \quad cy = y_{\min} + \frac{h}{2}$$

*(Do not execute remapping or retraining until user review and approval.)*

