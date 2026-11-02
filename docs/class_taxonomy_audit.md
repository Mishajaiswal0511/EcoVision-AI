# EcoVision AI — Comprehensive Semantic & Class-Definition Audit

**Target Document:** [`docs/class_taxonomy_audit.md`](file:///d:/Projects/ml_project/472/docs/class_taxonomy_audit.md)  
**Date of Audit:** 2026-10-31 22:33:00  
**Auditor:** Antigravity Advanced Agentic Systems  
**Audited Taxonomy:** 6 Classes (`0: plastic`, `1: metal`, `2: cardboard`, `3: glass`, `4: paper`, `5: general_trash`)  
**Audited Assets:**
- Dataset Config: [`data/dataset_v4.yaml`](file:///d:/Projects/ml_project/472/data/dataset_v4.yaml)
- Mapping Logic: [`data/label_mapping.json`](file:///d:/Projects/ml_project/472/data/label_mapping.json)
- Disposal Guidance: [`docs/disposal_guide.json`](file:///d:/Projects/ml_project/472/docs/disposal_guide.json)
- Frontend Constants: [`frontend/src/constants/classes.js`](file:///d:/Projects/ml_project/472/frontend/src/constants/classes.js)
- Backend Services: [`backend/services/disposal.py`](file:///d:/Projects/ml_project/472/backend/services/disposal.py) & [`backend/services/inference.py`](file:///d:/Projects/ml_project/472/backend/services/inference.py)

---

## 1. Executive Summary & Core Disconnect

EcoVision AI aims to provide real-world waste segregation and disposal guidance. However, this audit reveals a **fundamental semantic conflict** between:
1. **The User-Facing Application Contract (UI & Disposal Guide):** Promised to identify real-world recyclables vs. non-recyclable general waste (e.g. greasy pizza boxes, soiled tissues, sanitary items, composite wrappers).
2. **The Actual Machine Learning Supervision (Training Data):** Trained on a strict material-origin mapping where almost all packaging items—regardless of contamination or recyclability—were forced into the 5 recyclable material classes, leaving `general_trash` as a residual dustbin consisting almost exclusively of **cigarette butts and tiny outdoor litter specks**.

```
┌─────────────────────────────────────────────────────────────┐
│                 THE ECOVISION AI TAXONOMY CONFLICT          │
├──────────────────────────────┬──────────────────────────────┤
│ What the Frontend Promises   │ What the Dataset Trains      │
│ (User Expectations)          │ (Model Reality)              │
├──────────────────────────────┼──────────────────────────────┤
│ Greasy pizza boxes → Trash   │ Pizza boxes → CARDBOARD      │
│ Soiled tissues → Trash       │ Tissues → PAPER              │
│ Crisp & snack packets → Trash│ Crisp packets → PLASTIC      │
│ Takeaway foam clamshells →   │ Foam food containers →       │
│   Trash                      │   PLASTIC                    │
│ Disposable plastic cutlery → │ Plastic utensils → PLASTIC   │
│   Trash                      │                              │
│ General Trash = Household    │ General Trash = Cigarette    │
│   non-recyclables            │   butts (54%) & litter (43%) │
└──────────────────────────────┴──────────────────────────────┘
```

This misalignment explains why the model suffers from low recall and high confusion on real-world general waste: **the application promises functional disposal classification, but the training data enforces raw material categorization.**

---

## 2. Answers to Specific Audit Questions

### Q1: What exactly qualifies as plastic, metal, cardboard, glass, paper, and general_trash?

Under the current implementation across backend, frontend, and dataset mapping:
- **`plastic` (Class 0):** Encompasses all synthetic polymers, rigid bottles, flexible films, shopping bags, plastic caps, tubs, squeeze tubes, styrofoam pieces, foam food containers, and snack/crisp packets.
- **`metal` (Class 1):** Encompasses drink cans (aluminum), food tins (tin/steel), metal caps, metal lids, pop tabs, aluminum foil, aerosol cans, and small scrap metal.
- **`cardboard` (Class 2):** Encompasses corrugated shipping cartons, cereal/dry packaging boxes, toilet paper cores, egg cartons, meal cartons, pizza boxes, and aseptic drink cartons (Tetra Pak).
- **`glass` (Class 3):** Encompasses transparent, green, and amber glass bottles, food jars, cosmetic jars, and broken container glass. (Window pane and mirror glass are excluded per disposal guide).
- **`paper` (Class 4):** Encompasses newspapers, magazines, office sheets, clean paper bags, notebooks, flyers, wrapping paper, paper cups, paper straws, and facial tissues.
- **`general_trash` (Class 5):** In the dataset, only 4 source categories qualify: discarded cigarette butts, unidentifiable outdoor litter fragments, nylon/plastic rope-strings, and aluminum blister packs. In the frontend, it is described as greasy boxes, soiled tissues, sanitary waste, and mixed non-recyclables.

---

### Q2: Which real-world objects are included or excluded from each class?

| Target Class | Included Objects in Dataset | Excluded / Ambiguous Objects |
|---|---|---|
| **`plastic`** | PET bottles, HDPE jugs, caps, grocery bags, cling film, foam cups, styrofoam clam-shells, crisp packets, squeeze tubes, plastic cutlery | Carded blister packs (mapped to `null`), footwear (mapped to `null`), medical blister packs (mapped to `general_trash`) |
| **`metal`** | Soda cans, soup tins, aluminum foil, beer bottle crowns, jar lids, pull tabs, aerosol cans | Batteries (mapped to `null` due to hazardous nature), electronics (excluded) |
| **`cardboard`** | Amazon shipping boxes, cereal boxes, egg crates, pizza boxes, drink cartons (Tetra Pak), meal boxes | Heavily grease-soaked cardboard (no distinction made from clean cardboard) |
| **`glass`** | Beverage bottles, pickle/jam jars, broken bottle fragments | Ceramic mugs, window glass, drinking tumblers, mirrors (disposal guide warns against them, but model cannot differentiate) |
| **`paper`** | Printing paper, newspaper, magazines, kraft bags, paper coffee cups, paper straws, tissues | Thermal receipts (non-recyclable BPA/BPS paper), wax-coated butcher paper |
| **`general_trash`** | Cigarette butts, tiny unidentified beach/street debris, ropes/strings, pharma blister packs | Food waste (mapped to `null`), shoes/textiles (mapped to `null`), batteries (mapped to `null`), sanitary products, composite pouches |

---

### Q3: Are cigarette butts, wrappers, composite packaging, foam containers, dirty food containers, and sanitary waste assigned consistently?

**Verdict: HIGHLY INCONSISTENT.**

1. **Cigarette Butts:**
   - Assigned to `general_trash` (Class 5). Represent 53.9% of all general_trash in the dataset.
   - *Consistency:* Consistent within the dataset, but visually microscopic ($10 \times 10$ pixels at $640 \times 640$).
2. **Wrappers & Snack Packets:**
   - Multi-material metalized plastic film (e.g. potato chip packets, candy bar wrappers) was assigned to **`plastic`** (`"Crisp packet": "plastic"`, `"Other plastic wrapper": "plastic"`).
   - *Problem:* In real-world recycling systems, metalized film cannot be mechanically recycled with PET/HDPE bottles and belongs in general waste / energy recovery.
3. **Composite Packaging (Tetra Pak & Drink Cartons):**
   - Aseptic drink cartons (75% paper, 20% plastic, 5% aluminum foil) were assigned to **`cardboard`** (`"Drink carton": "cardboard"`).
   - Carded blister packs (paper backing + plastic bubble) were dropped entirely (`"Carded blister pack": null`).
   - Pharma blister packs (plastic + aluminum foil) were assigned to **`general_trash`** (`"Aluminium blister pack": "general_trash"`).
   - *Problem:* Three distinct composite packaging types were handled in three completely different ways: one as cardboard, one as null, and one as general trash!
4. **Foam Containers (Expanded Polystyrene / Styrofoam):**
   - Assigned to **`plastic`** (`"Foam food container": "plastic"`, `"Foam cup": "plastic"`, `"Styrofoam piece": "plastic"`).
   - *Problem:* Municipal curbside recycling universally rejects Styrofoam food containers because food grease permeates the porous polystyrene beads, making them non-recyclable.
5. **Dirty Food Containers & Pizza Boxes:**
   - `Pizza box` was assigned to **`cardboard`**.
   - `Meal carton` was assigned to **`cardboard`**.
   - `Disposable food container` was assigned to **`plastic`**.
   - *Problem:* Directly contradicts `docs/disposal_guide.json` ("pizza boxes with grease go to general waste; do not include food-soiled plastic") and `frontend/src/constants/classes.js` ("greasy pizza boxes belong in General Trash").
6. **Sanitary Waste & Tissues:**
   - `Tissues` was assigned to **`paper`**.
   - `Plastic glooves` (latex/nitrile gloves) was assigned to **`plastic`**.
   - *Problem:* Used facial tissues, paper towels, and single-use cleaning gloves are sanitary bio-waste that must go to general waste. Paper recycling plants reject tissues because the fibers are too short to re-pulp and are bio-contaminated.

---

### Q4: Are any objects visually similar but assigned to different classes?

Yes, several severe cross-class visual ambiguities exist in the training supervision:

1. **Paper Cups vs. Plastic Cups:**
   - `Paper cup` $\rightarrow$ `paper` (Class 4)
   - `Disposable plastic cup` $\rightarrow$ `plastic` (Class 0)
   - *Ambiguity:* White single-use cold cups made of polypropylene look virtually identical to white paper cups with poly linings from more than 1 meter away.
2. **Cardboard Meal Cartons vs. Plastic Takeaway Containers:**
   - `Meal carton` $\rightarrow$ `cardboard` (Class 2)
   - `Disposable food container` $\rightarrow$ `plastic` (Class 0)
   - *Ambiguity:* Modern compostable or bleached takeout clamshells share identical geometric forms, closure tabs, and proportions.
3. **Aluminium Foil vs. Plastic Film / Blister Packs:**
   - `Aluminium foil` $\rightarrow$ `metal` (Class 1)
   - `Plastic film` $\rightarrow$ `plastic` (Class 0)
   - `Aluminium blister pack` $\rightarrow$ `general_trash` (Class 5)
   - *Ambiguity:* Wrinkled foil, silvery chip packet interiors, and crumpled blister packs exhibit near-identical specular reflection and irregular crinkled geometry.
4. **Tissues vs. Paper Sheets vs. White Plastic Bags:**
   - `Tissues` $\rightarrow$ `paper` (Class 4)
   - `Normal paper` $\rightarrow$ `paper` (Class 4)
   - `Single-use carrier bag` (crumpled white polyethylene) $\rightarrow$ `plastic` (Class 0)
   - *Ambiguity:* A crumpled white tissue, a crumpled paper receipt, and a crumpled white plastic bag share identical color, edge density, and amorphous shapes.
5. **Cigarette Butts vs. Small Waste Debris:**
   - `Cigarette` $\rightarrow$ `general_trash` (Class 5)
   - `Plastic straw` (broken/cut piece) $\rightarrow$ `plastic` (Class 0)
   - `Pop tab` $\rightarrow$ `metal` (Class 1)
   - `Metal bottle cap` $\rightarrow$ `metal` (Class 1)
   - *Ambiguity:* In low-resolution or distant outdoor scenes, a 10-pixel cigarette butt, a 10-pixel straw fragment, and a 10-pixel crushed bottle cap are visually indistinguishable.

---

### Q5: Are any objects that should be recyclable being assigned to general_trash, or vice versa?

| Object Category | Source Class | Current Target Class | Real-World Recyclability Status | Segregation Conflict |
|---|---|:---:|---|---|
| **Aluminium blister pack** | `Aluminium blister pack` | `general_trash` | Non-recyclable composite (mostly) | Correct in practice, but inconsistent with carded blister pack (`null`). |
| **Crisp packets** | `Crisp packet` | `plastic` | **Non-recyclable** in curbside bins | **False Recyclable:** User is told to put it in blue bin, contaminating plastic stream. |
| **Foam food containers** | `Foam food container` | `plastic` | **Non-recyclable** expanded polystyrene | **False Recyclable:** Polystyrene clamshells contaminate municipal plastic baling. |
| **Disposable plastic cutlery** | `Plastic utensils` | `plastic` | **Non-recyclable** in optical sorting plants | **False Recyclable:** Small rigid cutlery falls through sorting screens (MRF glass screens). |
| **Drink cartons (Tetra Pak)** | `Drink carton` | `cardboard` | Multi-layer aseptic composite | **Contamination Risk:** Requires dedicated hydrapulping; contaminates pure corrugated cardboard. |
| **Facial tissues & napkins** | `Tissues` | `paper` | **Non-recyclable** bio-waste | **False Recyclable:** Contaminates paper pulp with bio-fluids and short unrecyclable fibers. |
| **Greasy pizza boxes** | `Pizza box` | `cardboard` | **Non-recyclable** if oil-soaked | **False Recyclable:** Oil prevents paper fibers from binding during repulping. |
| **Cleaning & medical gloves** | `Plastic glooves` | `plastic` | **Non-recyclable** sanitary waste | **False Recyclable:** Nitrile/latex contaminates mechanical plastic recycling. |

---

### Q6: Does the current taxonomy match the disposal recommendations shown by the application?

**NO. There is a documented contradiction between the UI/disposal service and the dataset labels.**

#### Conflict A: Disposal Preparation Guidance vs. Ground Truth
- In [`docs/disposal_guide.json`](file:///d:/Projects/ml_project/472/docs/disposal_guide.json), the system instructs:
  > *"Do not include heavily soiled paper (pizza boxes with grease go to general waste)"*  
  > *"Do not include food-soiled plastic — dispose in general waste"*
- But if a user presents a greasy pizza box or a dirty takeout container to the camera, the model will output **`cardboard`** or **`plastic`** with high confidence, because 100% of pizza boxes and food containers in the training data are labeled `cardboard` and `plastic`!

#### Conflict B: Frontend Class Examples vs. Model Capability
- In [`frontend/src/constants/classes.js`](file:///d:/Projects/ml_project/472/frontend/src/constants/classes.js), the UI states:
  > *General Trash examples: "Greasy pizza boxes, soiled tissues, sanitary waste, mixed non-recyclables"*
- But in reality:
  - If the model detects a pizza box, it outputs `cardboard`.
  - If the model detects a tissue, it outputs `paper`.
  - The model has seen **zero examples** of sanitary waste.
  - The model has seen **zero examples** of mixed domestic waste.
  - It only recognizes cigarette butts and unidentifiable beach pebbles as `general_trash`.

---

### Q7: Which class-definition ambiguities could cause incorrect predictions?

1. **`Unlabeled litter` Ambiguity:**
   - 43.1% of `general_trash` in training data is "Unlabeled litter" from TACO.
   - Because this category contains unidentifiable scraps, it creates an ill-defined gradient in the feature space: the model cannot extract a consistent shape, color, or texture archetype for `general_trash`.
2. **Food Packaging Dual-Identity:**
   - A greasy paper wrapper could be predicted as `paper`, `plastic` (if coated), or `general_trash` (if soiled).
   - Because visual models cannot reliably smell or touch grease levels, an object's visual appearance defaults to its base material, conflicting with its true segregation destination.
3. **Material vs. Destination Semantics:**
   - `plastic`, `metal`, `cardboard`, `glass`, `paper` are **material descriptions**.
   - `general_trash` is a **disposal destination / municipal routing category**.
   - Combining five material-based classes with one disposal-based class creates a category error: an object made of plastic (e.g. a dirty styrofoam cup) is simultaneously **material: plastic** and **destination: general_trash**.

---

## 3. Comprehensive Mapping Table: TACO (59 Classes) to Target Taxonomy

Below is the verified mapping from [`data/label_mapping.json`](file:///d:/Projects/ml_project/472/data/label_mapping.json) with diagnostic commentary:

| # | Source Category Name (TACO) | Target Class (0-5) | Target Name | Conflict / Diagnostic Severity |
|:---:|---|:---:|---|---|
| 1 | `Aerosol` | 1 | `metal` | **Clean:** Aluminium / tinplate pressurized canister. |
| 2 | `Aluminium blister pack` | 5 | `general_trash` | **Inconsistent:** Composite plastic/foil; carded pack was dropped. |
| 3 | `Aluminium foil` | 1 | `metal` | **Ambiguous:** Thin foil crinkles like film; often food-contaminated. |
| 4 | `Battery` | `null` | *Excluded* | **Correct:** Hazardous e-waste; belongs in separate red bin. |
| 5 | `Broken glass` | 3 | `glass` | **Safety Risk:** Model detects glass; UI warns about cuts. |
| 6 | `Carded blister pack` | `null` | *Excluded* | **Inconsistent:** Excluded while aluminium blister pack was retained. |
| 7 | `Cigarette` | 5 | `general_trash` | **Scale Problem:** Microscopic (sub-10px); dominates general_trash (54%). |
| 8 | `Clear plastic bottle` | 0 | `plastic` | **Clean:** Core recyclable PET bottle. |
| 9 | `Corrugated carton` | 2 | `cardboard` | **Clean:** Standard cardboard shipping box. |
| 10 | `Crisp packet` | 0 | `plastic` | **Disposal Conflict:** Metalized film is non-recyclable curbside. |
| 11 | `Disposable food container` | 0 | `plastic` | **Disposal Conflict:** Food-soiled plastic belongs in general waste. |
| 12 | `Disposable plastic cup` | 0 | `plastic` | **Clean:** Standard PP / PET disposable drink cup. |
| 13 | `Drink can` | 1 | `metal` | **Clean:** Core recyclable aluminium can. |
| 14 | `Drink carton` | 2 | `cardboard` | **Disposal Conflict:** Tetra Pak composite; non-corrugated. |
| 15 | `Egg carton` | 2 | `cardboard` | **Clean:** Molded pulp carton (or polystyrene variant). |
| 16 | `Foam cup` | 0 | `plastic` | **Disposal Conflict:** Styrofoam is non-recyclable general waste. |
| 17 | `Foam food container` | 0 | `plastic` | **Disposal Conflict:** Takeout clamshell; non-recyclable general waste. |
| 18 | `Food Can` | 1 | `metal` | **Clean:** Steel tin can. |
| 19 | `Food waste` | `null` | *Excluded* | **Clean:** Organic wet waste; excluded from dry recyclables. |
| 20 | `Garbage bag` | 0 | `plastic` | **Ambiguous:** Empty bag is plastic film; full bag is general waste. |
| 21 | `Glass bottle` | 3 | `glass` | **Clean:** Core recyclable glass beverage bottle. |
| 22 | `Glass cup` | 3 | `glass` | **Ambiguous:** Drinking glassware has different melting point than container glass. |
| 23 | `Glass jar` | 3 | `glass` | **Clean:** Core recyclable glass food jar. |
| 24 | `Magazine paper` | 4 | `paper` | **Clean:** Glossy paper publications. |
| 25 | `Meal carton` | 2 | `cardboard` | **Disposal Conflict:** Takeout food boxes are oil-contaminated. |
| 26 | `Metal bottle cap` | 1 | `metal` | **Clean:** Crown caps / screw caps. |
| 27 | `Metal lid` | 1 | `metal` | **Clean:** Jar lids. |
| 28 | `Normal paper` | 4 | `paper` | **Clean:** Office copy paper, sheets. |
| 29 | `Other carton` | 2 | `cardboard` | **Clean:** Miscellaneous paperboard packaging. |
| 30 | `Other plastic` | 0 | `plastic` | **Clean:** Rigid or semi-rigid plastic packaging. |
| 31 | `Other plastic bottle` | 0 | `plastic` | **Clean:** Detergent / shampoo / chemical bottles. |
| 32 | `Other plastic container` | 0 | `plastic` | **Clean:** Plastic tubs, buckets, trays. |
| 33 | `Other plastic cup` | 0 | `plastic` | **Clean:** Plastic cups. |
| 34 | `Other plastic wrapper` | 0 | `plastic` | **Disposal Conflict:** Multi-layer snack wrappers are general waste. |
| 35 | `Paper bag` | 4 | `paper` | **Clean:** Kraft shopping bags. |
| 36 | `Paper cup` | 4 | `paper` | **Disposal Conflict:** Polyethylene lined cups are non-recyclable curbside. |
| 37 | `Paper straw` | 4 | `paper` | **Clean:** Drinking straws. |
| 38 | `Pizza box` | 2 | `cardboard` | **Severe UI Contradiction:** Stated as general trash in frontend and guide. |
| 39 | `Plastic bottle cap` | 0 | `plastic` | **Clean:** Small bottle caps. |
| 40 | `Plastic film` | 0 | `plastic` | **Clean:** Stretch film, pallet wrap, grocery bags. |
| 41 | `Plastic glooves` | 0 | `plastic` | **Disposal Conflict:** Latex / nitrile gloves are sanitary general waste. |
| 42 | `Plastic lid` | 0 | `plastic` | **Clean:** Cup lids, container lids. |
| 43 | `Plastic straw` | 0 | `plastic` | **Disposal Conflict:** Single-use straws are non-recyclable general waste. |
| 44 | `Plastic utensils` | 0 | `plastic` | **Disposal Conflict:** Single-use cutlery is non-recyclable curbside. |
| 45 | `Polypropylene bag` | 0 | `plastic` | **Clean:** Woven sacks, PP pouches. |
| 46 | `Pop tab` | 1 | `metal` | **Clean:** Can ring pulls. |
| 47 | `Rope - strings` | 5 | `general_trash` | **Clean:** Entangling waste (tanglers) go to landfill. |
| 48 | `Scrap metal` | 1 | `metal` | **Clean:** Hardware, wire, fragments. |
| 49 | `Shoe` | `null` | *Excluded* | **Clean:** Footwear excluded. |
| 50 | `Single-use carrier bag` | 0 | `plastic` | **Clean:** Thin shopping bags. |
| 51 | `Six pack rings` | 0 | `plastic` | **Clean:** LDPE beverage rings. |
| 52 | `Spread tub` | 0 | `plastic` | **Clean:** Butter / margarine tubs. |
| 53 | `Squeezable tube` | 0 | `plastic` | **Disposal Conflict:** Toothpaste/lotion multi-layer tubes are general waste. |
| 54 | `Styrofoam piece` | 0 | `plastic` | **Disposal Conflict:** Expanded polystyrene is non-recyclable general waste. |
| 55 | `Tissues` | 4 | `paper` | **Severe UI Contradiction:** Soiled tissues stated as general trash in UI. |
| 56 | `Toilet tube` | 2 | `cardboard` | **Clean:** Inner core tubes. |
| 57 | `Tupperware` | 0 | `plastic` | **Clean:** Reusable storage containers. |
| 58 | `Unlabeled litter` | 5 | `general_trash` | **Ambiguity:** Undefined fragments; visually ambiguous debris. |
| 59 | `Wrapping paper` | 4 | `paper` | **Clean:** Gift wrap paper. |

---

## 4. Recommended Taxonomy Clarifications

To resolve the divergence between user expectations and model training, three distinct operational approaches can be adopted:

### Option A: Strict Material Taxonomy (Pure Object Detection)
- **Concept:** Re-label `general_trash` as **`litter_debris`** or **`other_waste`**.
- **Rule:** The model detects purely what an object is made of (`plastic`, `metal`, `cardboard`, `glass`, `paper`, `litter_debris`).
- **Application Logic:** The application backend uses secondary heuristics (e.g. asking the user "Is it greasy?" or providing conditional rules) to advise disposal:
  - *Example:* "Cardboard detected. If clean $\rightarrow$ Blue Bin. If soaked with grease $\rightarrow$ Black Bin."
- **Pros:** Keeps CV task visually clean and aligned with ground-truth material appearance.
- **Cons:** Places segregation burden on the user rather than the AI.

### Option B: Destination-Based Municipal Taxonomy (Actionable Segregation)
- **Concept:** Align classes with real-world municipal bins:
  1. `recyclable_plastic` (Clean rigid/film plastics)
  2. `recyclable_metal` (Clean cans, tins, foil)
  3. `recyclable_paper_cardboard` (Clean dry paper & corrugated cardboard)
  4. `recyclable_glass` (Bottles & jars)
  5. `general_waste` (Expanded to include: Styrofoam, crisp packets, greasy boxes, tissues, composite pouches, cigarette butts)
- **Pros:** 100% aligned with Indian and global municipal solid waste (MSW) bin colors and segregation rules.
- **Cons:** Requires re-annotating or re-mapping ~2,500 training instances and supplementing macro general waste images.

### Option C: Practical Hybrid (Align Definitions Without Retagging Everything)
- **Concept:** Retain the current 6 class IDs (`0..5`), but make two targeted corrections:
  1. **Update Frontend UI & Disposal Text:** Remove misleading promises from the UI. Change frontend description of `general_trash` from *"Greasy pizza boxes, soiled tissues"* to *"Cigarette butts, outdoor litter fragments, composite blister packs, non-recyclable debris"*.
  2. **Supplement Macro General Waste in Training Data:** Add genuine macro general waste images (dirty containers, styrofoam clamshells, packaging pouches) to `general_trash` so the class is not 97% microscopic cigarette butts.

---

## 5. Strategic Decisions Requiring User Approval

Before any training configuration or dataset modification is designed, the following key engineering decisions require your approval:

1. **Class 5 Semantic Scope:**
   - Should `general_trash` continue to be defined as microscopic outdoor litter (cigarette butts + ambiguous fragments)?
   - OR should it be expanded to represent real-world domestic non-recyclable trash (snack wrappers, styrofoam, soiled containers, tissues)?
2. **Food-Contaminated Items (Pizza Boxes & Takeaway Clamshells):**
   - Should pizza boxes remain classified as `cardboard` (material classification)?
   - OR should heavily soiled containers be re-routed to `general_trash` (functional disposal classification)?
3. **Frontend & Disposal Guide Alignment:**
   - Do you approve updating `frontend/src/constants/classes.js` and `docs/disposal_guide.json` to accurately reflect whatever model training policy is finalized, ensuring zero contradiction between the camera display and model output?
4. **Snack Packets & Styrofoam Re-Mapping:**
   - Should `Crisp packet` and `Foam food container` remain mapped to `plastic` (material ground truth)?
   - OR should they be re-mapped to `general_trash` to reflect their true non-recyclable curbside status?

---

> [!IMPORTANT]
> **Safety & Integrity Confirmed:**
> This audit was strictly read-only. No image files, label files, dataset manifests, model weights, or application source files were modified. Production weights [`models/weights_best.pt`](file:///d:/Projects/ml_project/472/models/weights_best.pt) remain untouched (`SHA-256: 1933635e9c9ca2c26c68204cc8ceae7950609386ddcc394a6731185bef694d87`).

