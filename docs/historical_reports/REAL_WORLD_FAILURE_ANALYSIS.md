# EcoVision AI — Comprehensive Real-World Failure Analysis

**Document:** `REAL_WORLD_FAILURE_ANALYSIS.md`  
**Date:** 2026-11-01  
**Author:** Computer Vision & ML Engineering Team  
**Scope:** Root-cause investigation of actual failure modes across real-world test scenes in EcoVision AI.

---

## 1. Executive Summary

A robust computer vision system must not conceal failure modes or rely on cosmetic workarounds. This investigation provides forensic evidence and root-cause categorization for key real-world detection failures observed on the EcoVision AI YOLO11s V5 candidate (`yolo11s_v5_best.pt` @ 800×800) and previous iterations (M1, M2, M3).

### Summary of Investigated Failure Modes:
| ID | Failure Mode | Test Image Asset | Symptom / Measured Metric | Primary Root Cause Category |
| :---: | :--- | :--- | :--- | :--- |
| **A** | **Cardboard Pile Clutter** | `cardboard_03.png` | 10 overlapping boxes, duplicate detections, low confidence (18–22%) | **Scale-Asymmetric NMS Failure & Training Ambiguity** |
| **B** | **Broken Glass Duplication & Ghost Metal** | `glass_02.webp` | 8 glass boxes on shards, 1 ghost metal box (conf 24.6%) inside bottle neck | **High-Frequency Shard Clustering & Containment NMS Blind Spot** |
| **C** | **Green Plastic Missed Detection** | `plastic_04.jpg` | 0 detections at 800px (conf 0.1217 filtered out); 16 detections at 1024px (conf up to 0.470) | **Feature Resolution & Convolutional Stride Limitation** |
| **D** | **Banana Peel Misclassification** | `organic_01.webp` | Classified as Glass (conf 59.8%) and Plastic (conf 28.4%) | **Severe Out-of-Distribution Training Omission** |
| **E** | **Garbage Heap Boundary Conflicts** | `garbage-heap...webp` | Low recall (14.3%), boundary misalignment, low confidence (15–35%) | **Extreme Intra-Class Variance & Amorphous Boundaries** |

---

## 2. Deep Root-Cause Analysis by Case

---

### Case A: Cardboard Pile Clutter (`cardboard_03.png`)

#### 1. Forensic Evidence
- **Input Image Size:** $497\times 505$ px.
- **Model Output @ 800px (`conf=0.15`):** 10 detections.
  - Box 0: `[33.8, 132.4, 278.4, 280.3]`, class: `cardboard`, conf: **0.510**
  - Box 1: `[7.7, 161.7, 432.2, 479.9]`, class: `cardboard`, conf: **0.306** (Area: $135,150$ px)
  - Box 2: `[246.8, 338.7, 441.1, 499.8]`, class: `cardboard`, conf: **0.225**
  - Box 3: `[2.5, 225.9, 136.9, 444.3]`, class: `cardboard`, conf: **0.225**
  - Box 4: `[21.9, 246.4, 409.1, 464.3]`, class: `cardboard`, conf: **0.221**
  - Box 5: `[17.4, 53.1, 113.4, 139.1]`, class: `cardboard`, conf: **0.197**
  - Box 7: `[385.5, 352.7, 428.3, 423.4]`, class: `plastic`, conf: **0.179** (False positive on tape)
  - Box 8: `[19.6, 161.7, 377.7, 326.4]`, class: `cardboard`, conf: **0.179**

#### 2. Root Cause Breakdown
1. **Scale-Asymmetric NMS Blind Spot:**
   - Standard Non-Maximum Suppression calculates:
     $$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
   - When Box 1 encompasses the entire pile ($425\times 318$ px) and Box 2 is a single nested box inside it ($194\times 161$ px):
     $$\text{Containment}(B_2 \subset B_1) = 83.6\%$$
     $$\text{IoU}(B_1, B_2) = \frac{31,234}{166,384} = \mathbf{0.187}$$
   - Because $0.187 < 0.45$ (default NMS IoU), standard NMS treats them as non-overlapping independent objects.
2. **Annotation Ambiguity in Training Set:**
   - In public waste datasets (such as TACO and Roboflow waste sets), annotators inconsistently label piles: some annotators draw one giant box around the entire pile, while others label each constituent box flap.
   - Consequently, the model learns bimodal predictions: it activates both for "aggregate cardboard heap" and "individual cardboard item".
3. **Low Confidence (18–22%):**
   - The detector penalizes overlapping proposal scores because the ground-truth distribution penalizes duplicate anchor assignments during Hungarian/Task-Aligned assignment.

#### 3. Corrective Actions
- **Post-Processing:** Implement **Containment-Aware Suppression (CAS)**. When $B_{\text{small}} \subset B_{\text{large}}$ with $\text{Containment} \ge 80\%$, suppress the aggregate cluster box if multiple cohesive sub-boxes exist, or suppress the redundant sub-box if it represents an unresolvable fragment.
- **Dataset Curation:** Enforce strict instance-level annotation rules in V5.1 (never label an entire pallet/pile as one box if sub-boxes are distinguishable).

---

### Case B: Broken Glass Duplication & Ghost Metal (`glass_02.webp`)

#### 1. Forensic Evidence
- **Input Image Size:** $500\times 500$ px.
- **Model Output @ 800px (`conf=0.15`):** 8 detections.
  - Box 0: `[17.0, 50.4, 110.9, 400.7]`, class: `glass`, conf: **0.913** (Main bottle body)
  - Box 1: `[127.1, 320.0, 184.8, 383.0]`, class: `glass`, conf: **0.454** (Shard cluster)
  - Box 2: `[201.4, 394.1, 245.3, 449.9]`, class: `glass`, conf: **0.426** (Small shard)
  - Box 3: `[241.9, 321.3, 396.5, 427.4]`, class: `glass`, conf: **0.421** (Large shard)
  - Box 6: `[38.4, 50.4, 78.2, 66.7]`, class: `metal`, conf: **0.246** (Ghost metal on bottle lip)

#### 2. Root Cause Breakdown
1. **Ghost Metal False Positive:**
   - Box 6 ($38\times 16$ px) covers the threaded glass neck / crown of the bottle.
   - The cylindrical threaded ridges visually mimic an aluminum screw cap or crown cork.
   - Box 6 is **100% contained** inside the main Glass bottle body (Box 0, $cont = 1.000$), but its area is small ($640$ px vs $32,994$ px), resulting in $\text{IoU} = 0.0194 \ll 0.45$.
   - Standard class-agnostic NMS cannot eliminate it because the IoU is under the threshold.
2. **Shard Fragmentation:**
   - Shattered glass produces dozens of sharp, irregular specular facets.
   - The model accurately detects genuine shards, but adjacent shard boxes partially overlap where shards touch on the ground.
3. **UI Text Collision:**
   - In the frontend canvas, Box 0 and Box 6 both originate near $y=50$ px. Rendering text badges at $(x_1, y_1)$ causes the label `"Metal 25%"` to directly overwrite `"Glass 91%"`.

#### 3. Corrective Actions
- **Containment Rule for Glass:** Any small low-confidence non-glass proposal ($conf < 0.30$) completely contained ($cont > 0.85$) inside a high-confidence glass container ($conf > 0.80$) must be suppressed unless visually detached.
- **Frontend Anti-Collision Layout:** Offset overlapping label pills vertically or clamp badges to non-occluding anchor corners.

---

### Case C: Green Plastic Object Missed (`plastic_04.jpg`)

#### 1. Forensic Evidence
- **Input Image Size:** $612\times 408$ px.
- **Model Output @ 640px (`conf=0.15`):** **0 detections**.
- **Model Output @ 800px (`conf=0.15`):** **0 detections** (Top raw proposal: `plastic` @ conf **0.1217**).
- **Model Output @ 1024px (`conf=0.15`):** **16 detections** (Top raw proposal: `plastic` @ conf **0.470**).

#### 2. Root Cause Breakdown
1. **Convolutional Stride & Spatial Receptive Field:**
   - YOLO11 uses downsampling strides of 8 (P3), 16 (P4), and 32 (P5).
   - In a $612\times 408$ image scaled down to fit within an 800px square with letterbox padding, fine plastic edges occupy fewer than $12$ pixels on the P3 head.
   - At 1024px, the feature resolution expands by $+64\%$ in area, allowing the P3 feature maps to capture distinctive contours and color transitions.
2. **Threshold Clipping:**
   - At 800px, the detector had in fact identified the object as `plastic` at $12.17\%$ confidence.
   - The static operational threshold of $15\%$ filtered out the true positive.

#### 3. Corrective Actions
- **Multi-Scale / High-Res Inference Option:** Offer a high-resolution toggle ($1024$ px) for detailed or fine-object scenes.
- **Adaptive Confidence Thresholding:** Provide a validated slider down to $0.10$ with UI guidance on false positive trade-offs.

---

### Case D: Banana Peel Misclassification (`organic_01.webp`)

#### 1. Forensic Evidence
- **Input Image Size:** $390\times 280$ px.
- **Model Output @ 800px (`conf=0.15`):**
  - Detection 1: `glass` @ conf **0.598**
  - Detection 2: `glass` @ conf **0.413**
  - Detection 3: `plastic` @ conf **0.284**
  - Class `organic`: **0 detections** (max activation $< 0.05$).

#### 2. Root Cause Breakdown
1. **Severe Out-of-Distribution Data Omission:**
   - As audited in Phase 2, `data/processed_v5/` contains 2,766 Organic annotations, but **100% of them originate from only 133 scenes** from Garbage in the Wild (GWD).
   - Every single training image depicts clustered, rotten scrap trays in outdoor compost dumps.
   - **Zero images** of isolated fruit skins (banana peels, orange peels, apple cores) on indoor tables or kitchen counters exist in the training set.
2. **Specular Feature Activation:**
   - The smooth, curved skin of a banana reflects indoor ambient lighting, producing specular highlights.
   - Because the training set contains thousands of smooth glass bottles and shiny plastic jugs, the model's convolutional kernels match the curvature and highlight gradients to `glass`.
3. **Ensemble Inability:**
   - Testing confirmed that ensembling M1, M2, M3, and M4 produces the identical error (`glass: 0.60`) because all models share the same training blind spot.

#### 3. Corrective Actions
- **Dataset Expansion (Tier 2 / V5.1):** Genuine resolution requires annotating $\ge 300$ real-world photos of isolated kitchen peels (bananas, citrus, vegetables) under diverse indoor and outdoor illumination.

---

### Case E: Garbage Pile Detection Degradation

#### 1. Forensic Evidence
- **Test Image Asset:** Outdoor garbage heap.
- **Model Output:** Low detection confidence (15–35%), fragmented bounding boxes, low General Trash recall.

#### 2. Root Cause Breakdown
1. **Intra-Class Variance:**
   - `general_trash` includes metallized snack wrappers, styrofoam clamshells, cigarette butts, dirty textiles, and torn composites.
   - The class has an mAP@50 of **0.1153** and recall of **14.3%** on the validation set.
2. **Amorphous Boundaries:**
   - Unlike rigid rectangular cardboard boxes or cylindrical bottles, litter heaps lack defined corners. Annotators draw subjective bounding boxes around arbitrary clusters, introducing label noise.

#### 3. Corrective Actions
- Focus V5.1 annotation guidelines on discrete litter items (snack bags, styrofoam cups) rather than diffuse piles.

---

## 3. Systematic Summary Matrix

| Failure Mode | Data Defect | Model Defect | Post-Processing Defect | UI Defect | Corrective Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cardboard Pile** | Mixed heap vs instance labels | Bimodal anchor firing | Scale-asymmetric IoU blind spot | Overlapping labels | Containment-Aware NMS + instance dataset rules |
| **Broken Glass** | Limited shard examples | Fragment confusion | IoU fails on small nested boxes | Label text collision | Glass-specific containment filtering + UI anti-collision |
| **Green Plastic** | Low colored plastic diversity | Feature map downsampling | Static 0.15 threshold cut | N/A | High-resolution mode (1024px) + threshold slider |
| **Banana Peel** | Complete absence of isolated peels | Texture/specular confusion | Cannot fix missing training data | N/A | Mandatory Tier 2 dataset expansion (V5.1) |
| **Garbage Pile** | Subjective cluster annotations | Low feature saliency | Disagrees on spatial boundaries | N/A | Discrete litter annotation rules + adaptive threshold |

