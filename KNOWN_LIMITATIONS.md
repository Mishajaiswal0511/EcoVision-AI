# EcoVision AI — Known Limitations & Real-World Edge Cases

**Project:** EcoVision AI (Autonomous Waste Detection & Segregation Guidance)  
**Candidate Checkpoint:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt` (M4, YOLO11s @ 800×800)  
**Evaluation Scope:** Academic Demonstration & Pre-Production Staging Verification  
**Status:** Documented for Academic Transparency (Zero Cosmetic Masking)

---

## 1. Executive Statement

In real-world waste detection, vision models face complex environmental conditions, varied lighting, severe occlusion, and extreme material diversity. EcoVision AI V5 represents a substantial architectural step forward over V4 by introducing a 7-class disposal-oriented taxonomy (`organic` class added) and high-resolution inference ($800\times 800$). 

However, rigorous testing against real-world test scenes has identified specific structural limitations that must be transparently understood and disclosed during academic demonstration.

---

## 2. Documented System Limitations

### Limitation 1: Banana Peel Misclassification (Organic vs. Glass / Plastic)
* **Observed Failure:** When presented with a standalone banana peel on a surface, the V5 candidate model predicts `glass` (~60% confidence) and low-confidence `plastic` (~11–18%), failing to recognize it as `organic`.
* **Root Cause Analysis:** 
  1. *Scene Clustering:* While the V5 training dataset contains 2,766 Organic annotations, they originate from only 133 distinct scenes (primarily dense scrap trays from the Garbage in the Wild dataset).
  2. *Data Blind Spot:* The dataset lacks standalone, isolated fruit peels (such as yellow/brown banana skins) on neutral household surfaces.
  3. *Specular Texture Confusion:* The glossy, smooth curvature of a banana skin produces specular reflections that the convolutional/attention backbones associate with smooth container glass or translucent plastic film.
* **Academic Demo Note:** Multi-model ensembling does **not** resolve this issue because all available checkpoints (V4 and V5) share the identical training gap. The genuine fix requires Tier 2 dataset enrichment (annotating isolated kitchen peels).

### Limitation 2: Weak Detection Recall on General Trash
* **Observed Failure:** In amorphous garbage piles or messy litter scenes, many items are missed, and confidence scores for `general_trash` remain predominantly low (0.15–0.35).
* **Root Cause Analysis:**
  1. *Extreme Intra-Class Variance:* `general_trash` represents a residual category comprising metallized snack packets (BOPP), polystyrene foam, cigarette butts, soiled napkins, and composite wrappers.
  2. *Benchmark Performance:* Even with Experiment B's $+48.6\%$ improvement over baseline, `general_trash` achieves an mAP@50 of **0.1153** and a recall of **14.3%** at operational threshold (`conf=0.15`).
  3. *Boundary Ambiguity:* Crumpled, torn, or semi-buried trash lacks rigid geometric boundaries, leading to low bounding box intersection and suppression.

### Limitation 3: Small and Heavily Overlapping Waste Objects
* **Observed Failure:** Small items (such as bottle caps, pull tabs, and torn fragments) and items in dense clusters are frequently missed or partially localized.
* **Root Cause Analysis:**
  1. At 800px input resolution, tiny objects occupy very few pixels (e.g., $< 16\times 16$ px), below the effective receptive field of P4/P5 feature pyramid heads.
  2. Severe occlusion causes boundary overlap. While `agnostic_nms=True` eliminates cross-class duplicate boxes on identical items, objects with low IoU ($< 0.45$) can either escape suppression or merge incorrectly when closely stacked.

### Limitation 4: Confidence Scores Are Not Calibrated Probabilities
* **Observed Reality:** The confidence scores reported by the YOLO detector (e.g., 60% Glass on a banana peel) are **heuristic model activation scores**, not calibrated statistical probabilities of real-world correctness.
* **Implications:**
  1. A confidence score of 60% does not guarantee a 60% chance of the object physically being glass.
  2. Neural detectors trained on imbalanced datasets frequently display overconfidence on majority visual features and underconfidence on rare or out-of-distribution classes.
  3. Academic presenters and users must interpret confidence scores as relative ranking metrics rather than absolute ground-truth certainty.

---

## 3. Transparency & Mitigation Guidelines for Demo

| Observed Issue | Why It Occurs | Demo Mitigation / Explanation |
| :--- | :--- | :--- |
| **Banana peel classified as Glass** | Training set lacks isolated fruit peels; surface curvature resembles glass bottle reflections. | Explain training distribution limits; highlight that Tier 2 dataset expansion is the proper ML engineering remedy. |
| **Low confidence on litter pile** | Debris is amorphous; general trash class has massive intra-class variance. | Demonstrate threshold slider: lowering `conf` from 0.25 to 0.15 surfaces additional litter items. |
| **Duplicate broken glass boxes** | Irregular shard edges produce multiple localized features on a single broken vessel. | Demonstrate `agnostic_nms=True` toggle: suppresses multi-class ghost boxes (e.g. ghost Metal box). |
| **Takeaway packaging ambiguity** | Detector sees clean plastic/cardboard shape; cannot detect food grease chemically. | Highlight EcoVision AI's **conditional disposal instructions** (clean lid $\to$ Blue Bin, soiled base $\to$ Grey Bin). |

---

## 4. Conclusion

These limitations are characteristic of real-world computer vision deployments and provide valuable discussion points for an academic defense. They validate the importance of rigorous error analysis, statutory disposal safeguards, and continuous dataset lifecycle management.

