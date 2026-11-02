# EcoVision AI — Conditional Packaging Remapping Audit

**Audit Date:** 2026-11-01  
**Scope:** Investigation of conditional packaging categories mapped into Class 5 (`general_trash`) during V5 dataset construction.  
**Focus Categories:** Pizza boxes, Meal cartons, and Disposable food containers.  
**Governing Standard:** MoEFCC / CPCB Solid Waste Management Rules 2026 & Bureau of Indian Standards (BIS) Recycling Norms.  
**Status:** **AUDITED & MITIGATED IN APPLICATION GUIDANCE**  
**Audit Verdict:** **PASS WITH STAGED POLICY MITIGATION**

---

## 1. Inventory of Affected Mappings

During the V4 $\to$ V5 taxonomy restructuring, three categories with food-contact potential were evaluated in [`models/experiments/v5_taxonomy/mapping_review.csv`](file:///d:/Projects/ml_project/472/models/experiments/v5_taxonomy/mapping_review.csv):

| Source Category Name | Dataset Origin | Instance Count | Former V4 Class | V5 Target Class | Mapping Type | Review Required | Stated Rationale |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `Pizza box` | TACO | **5** | `cardboard` | `general_trash` | CONDITIONAL | YES | Greasy food-soiled pizza boxes must go to general_trash per MoEFCC SWM rules. |
| `Meal carton` | TACO | **57** | `cardboard` | `general_trash` | CONDITIONAL | YES | Food-soiled takeout cartons must go to general_trash if oil-stained. |
| `Disposable food container` | TACO | **62** | `plastic` | `general_trash` | CONDITIONAL | YES | Visibly food-soiled takeout containers require residual disposal. |
| **TOTAL** | — | **124** | — | — | — | — | — |

---

## 2. Forensic Audit Findings: Evidence vs. Assumptions

### A. Blanket Assumption vs. Visual Evidence
- **Audit Finding:** In `scripts/package_v5_dataset.py` (lines 96–98), `Pizza box`, `Meal carton`, and `Disposable food container` were mapped deterministically at the category level into Class 5 (`general_trash`).
- **Contextual Justification:** TACO is an outdoor environmental litter dataset collected in streets, parks, beaches, and garbage dumps. Discarded takeaway containers in these environments are overwhelmingly contaminated with food residue, dirt, or grease.
- **Empirical Gap:** While the broad heuristic was reasonable given the outdoor litter collection domain, **individual images were not hand-inspected for localized grease spotting**. Consequently, if a clean, unsoiled pizza box lid or a spotless takeaway container was photographed, it was nonetheless assigned label ID 5 (`general_trash`).

### B. Physical and Industrial Recycling Reality
Under Indian municipal and industrial recycling standards:
1. **Paper & Cardboard (Hydrapulping Mills):** Cardboard is mixed with water in high-speed hydrapulpers to separate cellulose fibers. Vegetable oils and animal fats do not dissolve in water; they adhere to the fiber web, forming oily slicks, weak structural spots, and smoke during sheet drying. Therefore, **greasy pizza boxes genuinely ruin recycling batches**. Clean pizza box tops, however, can be cleanly recycled.
2. **Rigid Plastics (Flake Washing Lines):** Plastic containers with minor residues can be hot-washed with caustic soda in industrial wash lines. However, non-rinsed takeaway containers coated in coagulated grease or curries contaminate baled plastics and are rejected at Material Recovery Facilities (MRFs) as residual waste.

---

## 3. Strict Safety & Non-Disruption Guarantee

In strict compliance with Master Safety Rules:
1. **Zero Label Mutations:** No class IDs or ground-truth annotations in `data/processed_v5/` were modified or deleted.
2. **Zero Checkpoint Disruption:** The winning candidate checkpoint `yolo11s_v5_best.pt` was **NOT retrained or altered**.
3. **Zero Impact on Sealed Data:** `data/real_world_eval/` was not referenced or accessed.

---

## 4. Policy Mitigation in Application Staging

Because a single bounding box detector cannot determine invisible grease residues or internal food stains, the ambiguity is solved through **statutory conditional user instructions** in the disposal guidance layer:

1. **Cardboard Guidance ([`disposal_guide.json`](file:///d:/Projects/ml_project/472/models/experiments/v5_taxonomy/application_staging/disposal_guide.json)):**
   > *"CONDITIONAL RULE FOR PIZZA BOXES & MEAL CARTONS: Clean dry cardboard lids with zero oil can be recycled in Blue Bin. Portions saturated with grease, cheese, or food oils cannot be hydrapulped and must go to General Trash (Black/Grey Bin)."*
2. **Plastic Guidance:**
   > *"CONDITIONAL RULE FOR TAKEAWAY CONTAINERS: Rigid plastic food containers are accepted in Blue Bin ONLY if rinsed completely clean of food grease. Heavily oil-caked containers or foam clamshells must go to General Trash."*
3. **General Trash Guidance:**
   > *"Includes heavily food-contaminated packaging: greasy pizza boxes and food-soiled meal cartons where oil/grease cannot be cleaned off."*

---

## 5. Recommendations for Future Dataset Iterations (V6)
1. In future dataset expansions, do not treat takeaway food packaging as a monolithic class.
2. Introduce a secondary attribute tag: `packaging_cleanliness: clean | soiled` to enable conditional classification heads.
3. For V5 pre-production, the application staging mitigation provides legally sound, operationally safe instructions to the end-user.

