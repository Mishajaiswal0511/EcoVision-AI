# EcoVision AI — Comprehensive Dataset Audit & Expansion Plan

**Document:** `DATASET_AUDIT_REPORT.md`  
**Dataset Under Audit:** `data/processed_v5/` (V5 7-Class Disposal-Oriented Standard)  
**Date:** 2026-11-01  
**Audit Protocol:** Rigorous Ground-Truth Hash Audit, Geometry Distribution Analysis, and Cross-Split Leakage Verification

---

## 1. Executive Dataset Summary

The EcoVision AI V5 dataset standardizes waste detection on a 7-class taxonomy aligned with the Government of India Solid Waste Management Rules 2026 (MoEFCC / CPCB):
`plastic`, `metal`, `cardboard`, `glass`, `paper`, `general_trash`, `organic`.

### Global Dataset Metrics:
* **Total Images:** **11,249**
* **Total Annotated Bounding Boxes:** **38,308**
* **Training Set:** 8,242 images (7,657 labeled + 585 empty background negatives) | 27,576 instances
* **Validation Set:** 3,007 images (2,792 labeled + 215 empty background negatives) | 10,732 instances
* **Scene Separation:** Scene-hash grouped; **0% train/validation parent scene overlap**.
* **Empty Background Ratio:** $7.1\%$ across both splits (guards against false alarms on clean countertops/floors).

---

## 2. Per-Class Instance & Image Distribution

| Class ID | Class Name | Total Instances | Train Instances | Val Instances | Train Images Containing Class | Val Images Containing Class | Total Images Containing Class |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **plastic** | **8,642** | 6,362 | 2,280 | 2,129 | 730 | 2,859 |
| **1** | **metal** | **6,545** | 4,883 | 1,662 | 1,673 | 593 | 2,266 |
| **2** | **cardboard** | **4,995** | 3,563 | 1,432 | 1,261 | 472 | 1,733 |
| **3** | **glass** | **8,142** | 5,908 | 2,234 | 2,038 | 761 | 2,799 |
| **4** | **paper** | **4,649** | 3,392 | 1,257 | 1,442 | 525 | 1,967 |
| **5** | **general_trash** | **2,569** | 1,926 | 643 | 760 | 246 | 1,006 |
| **6** | **organic** | **2,766** | 1,542 | 1,224 | **92** | **41** | **133** |
| **—** | **Empty Negatives** | **—** | — | — | 585 | 215 | 800 |
| **TOTAL** | | **38,308** | **27,576** | **10,732** | **8,242** | **3,007** | **11,249** |

---

## 3. Object Scale & Geometry Distribution

Following standard COCO scale criteria evaluated in 800×800 pixel space:
- **Small:** $\text{Area} < 32^2$ px ($< 1,024$ px$^2$)
- **Medium:** $32^2 \le \text{Area} < 96^2$ px ($1,024$ to $9,216$ px$^2$)
- **Large:** $\text{Area} \ge 96^2$ px ($\ge 9,216$ px$^2$)

### Training Set Scale Breakdown:
| Class | Small Instances | Medium Instances | Large Instances | % Small | % Large | Geometric Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **plastic** | 683 | 2,087 | 3,592 | 10.7% | 56.5% | Balanced; includes caps, tubs, and large bottles |
| **metal** | 328 | 1,378 | 3,177 | 6.7% | 65.1% | Predominantly beverage cans and large tins |
| **cardboard** | 44 | 380 | 3,139 | 1.2% | **88.1%** | Almost exclusively large shipping boxes and cartons |
| **glass** | 265 | 1,085 | 4,558 | 4.5% | **77.2%** | Dominated by full bottles and jars |
| **paper** | 49 | 477 | 2,866 | 1.4% | **84.5%** | Dominated by sheets, newspapers, bags |
| **general_trash** | **1,202** | 425 | 299 | **62.4%** | 15.5% | **Heavily skewed toward tiny litter fragments** |
| **organic** | 231 | 728 | 583 | 15.0% | 37.8% | Clustered scrap trays and dense compost chunks |

---

## 4. Key Diagnostic Findings & Dataset Blind Spots

### Finding 1: Extreme Clustering in Organic Waste
- While Organic boasts **2,766 bounding boxes**, they reside in only **133 total images** (an average of **20.8 instances per image**).
- All 133 images originate from the Garbage in the Wild (GWD) dataset's dense organic sorting trays.
- **The Blind Spot:** There are **zero individual, standalone kitchen food scraps** (e.g. single banana peels, apple cores, citrus rinds, carrot shavings) sitting on clean kitchen countertops, wood tables, or tile floors.
- **Consequence:** The model cannot generalize to single-object fruit peel detection in household settings, causing the notorious banana peel misclassification.

### Finding 2: General Trash Size & Representation Imbalance
- Over **62.4%** of general trash annotations in the training set are tiny ($< 32\times 32$ px) cigarette butts and shredded plastic/paper fragments from TACO.
- Only 299 instances (15.5%) are large residual waste items (e.g., takeaway clamshells, crumpled chip bags, greasy cartons).
- **Consequence:** Low recall and low confidence when detecting medium-to-large mixed trash objects in consumer waste bins.

### Finding 3: Cardboard Pile Instance Ambiguity
- Over **88%** of cardboard instances are labeled as giant bounding boxes.
- Multiple training sets contain bimodal annotations: some images outline each individual box, while others draw a single box over an entire pile of 10 boxes.
- **Consequence:** The detector suffers from overlapping bounding boxes and low confidence on cardboard piles because it cannot determine whether to predict the cluster or the individual boxes.

---

## 5. Prioritized Dataset Expansion Plan (V5.1)

To bring EcoVision AI to true industrial reliability, dataset expansion must directly target the identified blind spots:

### Phase 2.1: Target Collection Quotas for V5.1

| Target Category | Specific Object Types Required | Target New Images | Target New Instances | Collection Setting |
| :--- | :--- | :---: | :---: | :--- |
| **Organic (Peels & Scraps)** | Standalone yellow/brown banana peels, orange/lemon rinds, apple cores, potato skins, eggshells, tea bags | +300 | +600 | Clean countertops, wooden tables, tile floors, single-object framing |
| **General Trash (Packaging)** | Crushed metallized chip/crisp bags, styrofoam clamshells, plastic straws, disposable cutlery, grease-soaked pizza box corners | +250 | +500 | Trash bins, mixed waste baskets, outdoor litter bins |
| **Colored Plastic** | Opaque colored HDPE detergent bottles, green/blue plastic toys, colored plastic tubs, bottle caps | +200 | +400 | Kitchen surfaces, recycling bins |
| **Cardboard Piles** | Partially overlapping folded cartons, torn cardboard flaps, flattened Amazon boxes | +150 | +450 | Floor recycling corners, warehouse piles |
| **Broken Glass** | Shattered clear and amber glass shards, fractured bottle bases, cracked jars | +100 | +300 | Clean indoor surfaces, controlled safe arrangements |

### Phase 2.2: Sourcing & Licensing Standards
1. **Permitted Sources:**
   - Real-world original mobile camera captures (CC-BY or proprietary domain ownership).
   - Filtered subsets of TrashBox, TACO, and Roboflow Public Waste datasets under permissible licenses (CC-BY 4.0, MIT, Apache 2.0).
2. **Forbidden Practices:**
   - No synthetic cut-and-paste images without realistic shadow and lighting rendering.
   - Zero access or contamination from `data/real_world_eval/`.
   - Strict scene grouping: all newly captured images from the same camera session/environment must be assigned together to either Train or Val.

### Phase 2.3: Annotation Quality Rules
- **No Aggregate Bounding Boxes:** Every distinguishable object must have its own tight bounding box.
- **Occluded Objects:** If an object is $> 70\%$ occluded, do not label unless its identity is obvious from the visible portion.
- **Contaminated Packaging:** Soiled pizza boxes with visible grease must be annotated as `general_trash`; clean corrugated lids must be annotated as `cardboard`.

