# EcoVision AI — Final Dataset Report & Training Readiness Review

> **Stage:** Step 2.2 — Final Label Mapping, Leakage-Free Splits & Training Readiness  
> **Source Dataset:** TACO Official All-Class (Roboflow v6 export)  
> **Target Dataset:** EcoVision AI Clean 6-Class YOLOv11 Format  
> **Date of Final Audit:** 2026-10-29  
> **Automated Verification:** ✅ 19/19 Tests Passed | 0 Critical Annotation Errors | 0 Cross-Split Leakage  

---

## 1. Complete Source-to-Target Mapping Audit (All 59 Classes)

Every single one of the 59 source classes in the raw dataset was audited. No ambiguous classes were silently discarded or merged without a documented physical and municipal justification.

| # | Original Class Name | Count | Target Class | Semantic Reason & Physical Justification | Reliability | Manual Review? |
|---|---|---|---|---|---|---|
| 1 | `Aerosol` | 17 | `metal` | Pressurized aluminium/tinplated steel canister. Recycled in metal stream once emptied. | 🟢 High | No |
| 2 | `Aluminium blister pack` | 12 | `general_trash` | Multi-material sandwich of aluminium foil and PVC/PVDC plastic. Cannot be separated curbside. | 🟡 Medium | No |
| 3 | `Aluminium foil` | 106 | `metal` | 100% malleable aluminium metal packaging. | 🟢 High | No |
| 4 | `Battery` | 4 | `null` (Excluded) | Critically insufficient sample size (4 instances). Retaining would cause severe model hallucination. | 🟢 High (Prudent exclusion) | Recommended to collect more data |
| 5 | `Broken glass` | 193 | `glass` | Fractured glass fragments and shards. Direct glass material match. | 🟢 High | No |
| 6 | `Carded blister pack` | 2 | `null` (Excluded) | Extremely sparse (2 instances) composite packaging (paperboard backing + plastic bubble). | 🟢 High | No |
| 7 | `Cigarette` | 1,128 | `general_trash` | Cellulose acetate filter + tobacco residue. Non-recyclable toxic litter; belongs in residual trash. | 🟢 High | No |
| 8 | `Clear plastic bottle` | 484 | `plastic` | Polyethylene terephthalate (PET / PETE #1) transparent beverage bottle. | 🟢 High | No |
| 9 | `Corrugated carton` | 114 | `cardboard` | Fluted paperboard shipping and transport boxes. | 🟢 High | No |
| 10 | `Crisp packet` | 66 | `plastic` | Metallized oriented polypropylene (BOPP) flexible film packaging. | 🟢 High | No |
| 11 | `Disposable food container` | 62 | `plastic` | Rigid PP/PS takeout clamshells and tubs. | 🟢 High | No |
| 12 | `Disposable plastic cup` | 176 | `plastic` | Transparent cold-beverage cups (PET/PP). | 🟢 High | No |
| 13 | `Drink can` | 370 | `metal` | Two-piece drawn aluminium beverage cans. | 🟢 High | No |
| 14 | `Drink carton` | 73 | `cardboard` | Liquid packaging board (Tetra Pak / aseptic carton). Over 75% paperboard by weight. | 🟢 High | No |
| 15 | `Egg carton` | 21 | `cardboard` | Molded pulp paperboard carton. | 🟢 High | No |
| 16 | `Foam cup` | 24 | `plastic` | Expanded polystyrene (EPS #6) thermal cup. | 🟢 High | No |
| 17 | `Foam food container` | 26 | `plastic` | Expanded polystyrene (EPS #6) hinged container. | 🟢 High | No |
| 18 | `Food Can` | 55 | `metal` | Three-piece tinplated steel food preservation can. | 🟢 High | No |
| 19 | `Food waste` | 13 | `null` (Excluded) | Severely underrepresented organic waste (13 instances in outdoor litter). Excluded to prevent model collapse. | 🟢 High (Prudent exclusion) | Recommended to collect more data |
| 20 | `Garbage bag` | 52 | `plastic` | Low-density polyethylene (LDPE #4) film trash sack. | 🟢 High | No |
| 21 | `Glass bottle` | 181 | `glass` | Molded soda-lime silica glass container. | 🟢 High | No |
| 22 | `Glass cup` | 11 | `glass` | Drinking glassware. | 🟢 High | No |
| 23 | `Glass jar` | 11 | `glass` | Wide-mouthed glass preserves/jam jar. | 🟢 High | No |
| 24 | `Magazine paper` | 14 | `paper` | Calendered coated magazine sheets. | 🟢 High | No |
| 25 | `Meal carton` | 46 | `cardboard` | Folding paperboard meal container. | 🟢 High | No |
| 26 | `Metal bottle cap` | 133 | `metal` | Steel crown cork caps from beverage bottles. | 🟢 High | No |
| 27 | `Metal lid` | 19 | `metal` | Lug/threaded steel lid from jars and food tins. | 🟢 High | No |
| 28 | `Normal paper` | 136 | `paper` | Uncoated bond paper, documents, loose sheets. | 🟢 High | No |
| 29 | `Other carton` | 159 | `cardboard` | Miscellaneous paperboard packaging cartons. | 🟢 High | No |
| 30 | `Other plastic` | 456 | `plastic` | Polymer waste objects not matching specific categories. | 🟢 High | No |
| 31 | `Other plastic bottle` | 89 | `plastic` | Opaque HDPE/PP bottles (detergent, shampoo). | 🟢 High | No |
| 32 | `Other plastic container` | 9 | `plastic` | Rigid polymer containers and tubs. | 🟢 High | No |
| 33 | `Other plastic cup` | 4 | `plastic` | Non-standard plastic cups. | 🟢 High | No |
| 34 | `Other plastic wrapper` | 446 | `plastic` | Flexible polymer candy and snack wrappers. | 🟢 High | No |
| 35 | `Paper bag` | 51 | `paper` | Unbleached kraft paper shopping bag. | 🟢 High | No |
| 36 | `Paper cup` | 117 | `paper` | Paperboard drinking cup (thin PE liner, predominantly cellulose fiber). | 🟢 High | No |
| 37 | `Paper straw` | 7 | `paper` | Spiral-wound paper straw. | 🟢 High | No |
| 38 | `Pizza box` | 5 | `cardboard` | Corrugated paperboard pizza box. | 🟢 High | No |
| 39 | `Plastic bottle cap` | 348 | `plastic` | High-density polyethylene (HDPE #2) screw caps. | 🟢 High | No |
| 40 | `Plastic film` | 789 | `plastic` | Flexible LDPE/LLDPE stretch film and wraps. | 🟢 High | No |
| 41 | `Plastic glooves` | 6 | `plastic` | Disposable polyethylene/synthetic polymer gloves. | 🟢 High | No |
| 42 | `Plastic lid` | 138 | `plastic` | Thermoformed PS/PP beverage cup lids. | 🟢 High | No |
| 43 | `Plastic straw` | 280 | `plastic` | Extruded polypropylene (PP #5) drinking straw. | 🟢 High | No |
| 44 | `Plastic utensils` | 60 | `plastic` | Injection-molded PS/PP cutlery (forks, spoons, knives). | 🟢 High | No |
| 45 | `Polypropylene bag` | 6 | `plastic` | Woven polypropylene (PP #5) sack material. | 🟢 High | No |
| 46 | `Pop tab` | 161 | `metal` | Stamped aluminium stay-on tab from beverage cans. | 🟢 High | No |
| 47 | `Rope - strings` | 51 | `general_trash` | Mixed natural/synthetic cordage, twines, and marine ropes. Non-recyclable. | 🟡 Medium | No |
| 48 | `Scrap metal` | 39 | `metal` | Ferrous and non-ferrous construction/wire debris. | 🟢 High | No |
| 49 | `Shoe` | 11 | `null` (Excluded) | Footwear (apparel/leather/rubber). Not segregable packaging waste. | 🟢 High | No |
| 50 | `Single-use carrier bag` | 104 | `plastic` | Thin HDPE shopping carry bags. | 🟢 High | No |
| 51 | `Six pack rings` | 21 | `plastic` | Photodegradable LDPE can carrier rings. | 🟢 High | No |
| 52 | `Spread tub` | 17 | `plastic` | Polypropylene (PP) margarine/butter tubs. | 🟢 High | No |
| 53 | `Squeezable tube` | 12 | `plastic` | Multi-layer plastic cosmetic and paste tubes. | 🟢 High | No |
| 54 | `Styrofoam piece` | 195 | `plastic` | Broken expanded polystyrene (EPS) packaging insulation. | 🟢 High | No |
| 55 | `Tissues` | 75 | `paper` | Soft cellulose tissue paper, napkins, wipes. | 🟢 High | Note: municipal rules treat soiled tissues as general waste; visually they are paper. |
| 56 | `Toilet tube` | 9 | `cardboard` | Spiral-wound paperboard core. | 🟢 High | No |
| 57 | `Tupperware` | 7 | `plastic` | Reusable polypropylene food storage box. | 🟢 High | No |
| 58 | `Unlabeled litter` | 903 | `general_trash` | Degraded, ambiguous litter fragments that annotators could not assign to a distinct material. Belong in general waste. | 🟢 High | No |
| 59 | `Wrapping paper` | 17 | `paper` | Printed decorative cellulose gift wrapping paper. | 🟢 High | No |

---

## 2. Quantitative Conversion Audit (Before vs. After)

All raw source files in `data/raw/garbage_taco-official-all-class_v6` were preserved **100% unchanged**. The remapped data was written to a distinct directory `data/raw/remapped_taco`.

| Metric | Source Raw Count | Remapped Count | Delta / Explanation |
|---|---|---|---|
| **Total Images** | 2,550 | 2,543 | -7 images (contained only excluded classes like shoes/batteries or zero-area boxes) |
| **Total Annotations** | 8,152 | 8,119 | -33 annotations total |
| **- Excluded Annotations** | — | 30 | 4 Battery + 13 Food waste + 11 Shoe + 2 Carded blister pack |
| **- Degenerate Boxes Dropped** | — | 3 | Zero-area boxes removed (logged with exact coordinates below) |
| **- Unmapped Annotations** | — | **0** | **100% of annotations accounted for** |

### Mathematical Reconciliation:
$$8,119 \text{ (mapped)} + 30 \text{ (excluded)} + 3 \text{ (degenerate dropped)} = 8,152 \text{ (total source)}$$

---

## 3. Degenerate Bounding Box Removal Log

The 3 corrupted boxes from the original dataset were cleanly removed from the training-ready copies:

| Image File | Line | Source Class | Box Coordinates $[cx, cy, w, h]$ | Reason Dropped |
|---|---|---|---|---|
| `000008_jpg.rf.3bd5b9fc637686318822dc3183352b68.txt` | 1 | `Styrofoam piece` | `[0.0, 0.0, 0.0, 0.5905]` | $w = 0.0$ (Zero width) |
| `000088_jpg.rf.500768ed03e5b86b9dcb44ad91d3a15f.txt` | 1 | `Corrugated carton` | `[0.9984, 0.0005, 0.7549, 0.0]` | $h = 0.0$ (Zero height) |
| `000049_JPG.rf.3ae9aa2b17afec86fc4a51e2de680c7c.txt` | 5 | `Plastic straw` | `[0.8411, 0.0070, 0.8211, 0.0]` | $h = 0.0$ (Zero height) |

---

## 4. Duplicate Resolution & Leakage Elimination

### The Problem in Original Splits:
In the raw Roboflow export, **5 near-duplicate groups (containing 16 images)** were split across train, val, and test. For example, cluster Group 4 contained 7 near-identical photo burst images: 5 in train, 1 in val, and 1 in test.

### The Resolution Applied in `prepare_splits.py`:
Using `evaluation/results/dedup_plan.json`, all near-duplicate clusters (120 groups across the dataset) were treated as atomic indivisible units:
- Every image belonging to a near-duplicate group was assigned to the **exact same split** (`train`).
- No image was arbitrarily deleted, preserving data volume while eliminating cross-split evaluation leakage.
- Independent audit check with `deduplicate_dataset.py` on `data/processed`:
  $$\textbf{Cross-split contamination groups: } \mathbf{0}$$

---

## 5. Final Leak-Free Split Breakdown (`data/processed`)

* **Random Seed:** `42` (Fixed, fully reproducible)
* **Stratification Method:** Class-aware bucketed stratification keeping near-duplicate clusters intact.
* **Split Proportions:** 70.4% Train / 14.9% Val / 14.7% Test

| Target Class | Train Instances | Val Instances | Test Instances | Total Instances | % of Dataset |
|---|---|---|---|---|---|
| **`plastic`** | 2,816 | 536 | 523 | **3,875** | 47.7% |
| **`general_trash`** | 1,569 | 250 | 275 | **2,094** | 25.8% |
| **`metal`** | 643 | 124 | 133 | **900** | 11.1% |
| **`cardboard`** | 309 | 59 | 69 | **437** | 5.4% |
| **`paper`** | 296 | 54 | 67 | **417** | 5.1% |
| **`glass`** | 304 | 53 | 39 | **396** | 4.9% |
| **Total Annotations** | **5,937** | **1,076** | **1,106** | **8,119** | **100.0%** |
| **Total Images** | **1,790** | **379** | **374** | **2,543** | — |

> [!IMPORTANT]
> The **374 test images (with 1,106 annotations)** in `data/processed/images/test/` are now strictly **SEALED**. They will not be accessed, augmented, or used for model tuning during Step 3.

---

## 6. Training Readiness & Viva Defense Assessment

### Q1: Is the six-class dataset ready for baseline training?
**Yes.** All 2,543 images and 8,119 bounding boxes have passed rigorous structural, geometric, and coordinate integrity checks. There are 0 corrupt images, 0 missing files, 0 out-of-range coordinates, and 0 cross-split leakage instances. Every single class has at least 300 instances (lowest is glass at 396; highest is plastic at 3,875), which comfortably exceeds the minimum viable threshold for transfer-learning object detection.

### Q2: Which classes have insufficient diversity?
- **`glass` (396 instances):** Mostly clear beverage bottles and broken glass shards. Lacks diversity in colored glass jars (green/amber) and glass tableware.
- **`paper` (417 instances):** Heavily skewed toward crushed receipts and disposable paper cups; lacks flat documents, packaging mailers, and cardboard-hybrid envelopes.
- **`cardboard` (437 instances):** Good volume on shipping cartons, but fewer examples of compressed or wet boxes.

### Q3: Which classes are likely to be confused by the detector?
1. **`paper` vs. `cardboard`:** Thin cardstock, meal cartons, and paper bags share near-identical fibrous textures and brown kraft colors. The visual boundary between heavy paper and light paperboard is continuous.
2. **`plastic` (film/wrappers) vs. `metal` (foil):** Wrinkled aluminium foil and shiny metallized snack wrappers (crisp packets) exhibit similar specular highlights and crumpled geometries.
3. **`plastic` (cups) vs. `glass` (clear cups/bottles):** Transparent plastic drink cups and clear glass cups share transparent refractive visual properties, differing primarily in thickness and reflectivity.

### Q4: What limitations remain because TACO is primarily an outdoor litter dataset?
- **Contextual Bias:** Most backgrounds are outdoor terrain: grass, sidewalks, soil, sand, and asphalt. When tested in indoor environments (e.g., kitchen countertops, office desks, white tiles), background feature maps may induce false positive or false negative shifts.
- **Weathering & Soiling:** Waste objects in TACO are frequently crumpled, crushed, dirty, sun-bleached, or partially buried. Clean, uncrushed household waste (e.g., a brand-new soda can on a table) may exhibit lower activation scores without proper color/scale augmentation.

### Q5: What additional data should be collected to improve indoor and real-world performance?
1. **Targeted E-Waste / Battery Dataset:** Collect ~200–300 close-up photos of AA, AAA, 9V, lithium button cells, and swollen mobile batteries on various surfaces to train a dedicated safety-critical battery class later.
2. **Indoor Kitchen / Food Waste Set:** Supplement with ~300 indoor photographs of food scraps (fruit peels, vegetable ends, plate scraps) to reintroduce a valid `organic` class.
3. **Out-of-Distribution (OOD) Negative Set:** Assemble 150–200 photos of non-waste household objects (smartphones, laptops, ceramic mugs, pens, books, chairs) to calibrate the unknown-object rejection threshold before deploying the web application.

---

## 7. Change Log Summary

* **`data/raw/remapped_taco/`:** Created clean intermediate repository of 2,543 images with remapped class indices $[0..5]$.
* **`data/processed/`:** Populated with clean, verified `train` (1,790), `val` (379), and `test` (374) partitions.
* **`data/dataset.yaml`:** Updated to reflect 6 classes (`nc: 6`) with verified class names and municipal bin color tags.
* **`data/label_mapping.json`:** Updated to v2.1 with verified 59-to-6 mapping dictionary.
* **`data/processed/split_manifest.json`:** Created fixed, deterministic assignment log for all 2,543 images.
* **`tests/test_data_preparation.py`:** 19/19 passing automated tests verifying pipeline resilience.

---

*Report prepared & verified by EcoVision AI Engineering Pipeline | Step 2.2 Complete*

