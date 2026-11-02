# EcoVision AI — Pre-Training Dataset Quality Review

> **Audit Type:** Independent Pre-Training Quality & Integrity Review (Pre-Step 3)  
> **Target Status:** **VERIFIED PASS**  
> **Date:** 2026-10-29  
> **Auditor Role:** Senior Computer Vision Engineer & ML Researcher  
> **Scope:** Verification of Split Integrity, Taxonomy Reliability, Annotation Geometry, Mathematical Reconciliation, and Evaluation Readiness.

---

## 1. Overall Audit Verdict: **PASS**

Every empirical check on the filesystem, annotations, split manifests, and configurations has been verified. No model training has taken place. The dataset in `data/processed` is structurally sound, mathematically reconciled, and ready for baseline transfer learning with YOLOv11s.

---

## 2. Checkpoint 1: Split Integrity & Leakage Verification

### 2.1 The 5 Cross-Split Contamination Groups from Raw Data
In the raw Roboflow export, 5 photo-burst clusters (16 images total) crossed split boundaries. We inspected each group against `data/processed/split_manifest.json`:

| Group | Size | Original Roboflow Splits | Processed Split Assignment | Cross-Split Leakage? |
|---|---|---|---|---|
| **Group 1** | 2 | `['train', 'val']` | `{'train'}` | 🟢 **Zero Leakage** |
| **Group 2** | 2 | `['test', 'train']` | `{'train'}` | 🟢 **Zero Leakage** |
| **Group 3** | 2 | `['train', 'val']` | `{'train'}` | 🟢 **Zero Leakage** |
| **Group 4** | 7 | `['test', 'train', 'val']` | `{'train'}` | 🟢 **Zero Leakage** |
| **Group 5** | 3 | `['train', 'val']` | `{'train'}` | 🟢 **Zero Leakage** |

*Result:* All 5 contaminated clusters are now strictly isolated within the `train` partition. None cross into `val` or `test`.

### 2.2 Investigation of the 120 Near-Duplicate Clusters
* **Finding:** The 120 near-duplicate clusters identified by `pHash` (distance $\le 10$) were **not** all forced into `train`.
* **Empirical Split Breakdown:**
  * **`train`:** 91 clusters (75.8%)
  * **`val`:** 17 clusters (14.2%)
  * **`test`:** 12 clusters (10.0%)
* **Why this is optimal:** `prepare_splits.py` uses cluster-atomic bucketed stratification. Every image within a given cluster stays in the **same split**. Distributing clusters across train, val, and test (~75/14/10 ratio) preserves real-world scene diversity in validation and test while maintaining an airtight boundary between them.
* **Test Set Bias Assessment:** The test set contains 374 images and 1,106 annotations across all 6 classes. It was **not** starved or biased by the deduplication algorithm.
* **Verification Command:** `scripts/deduplicate_dataset.py` run on `data/processed` reported:
  $$\textbf{Cross-split contamination groups: } \mathbf{0}$$

---

## 3. Checkpoint 2: Label Mapping & Visual Learnability Review

We evaluated whether mapping 59 raw classes to 6 target classes creates visually inconsistent or contradictory training signals for a convolutional/attention backbone.

### 3.1 Review of Critical Object Categories

| Category | Raw Classes Included | Count | Target Class | Visual Learnability & Computer Vision Rationale |
|---|---|---|---|---|
| **Paper Cups** | `Paper cup` | 117 | `paper` | **Consistent.** Thin internal polyethylene moisture barrier is visually invisible. Feature maps learn cylindrical cup geometry with paper fiber matte texture. |
| **Drink Cartons** | `Drink carton` | 73 | `cardboard` | **Consistent.** Aseptic composite cartons (Tetra Pak) have a boxy paperboard structure. Treated as carton/cardboard in municipal collection. |
| **Aluminium Foil** | `Aluminium foil` | 106 | `metal` | **Consistent.** Highly reflective metallic sheen and crumpled creases provide distinct specular features for metal. |
| **Metallized Wrappers** | `Crisp packet` | 66 | `plastic` | **Consistent.** High-gloss polymer multi-layer packaging. Visually aligns with flexible plastics. |
| **Bottle Caps** | `Plastic bottle cap`<br>`Metal bottle cap` | 348<br>133 | `plastic`<br>`metal` | **Visually Distinguishable.** Plastic caps are threaded cylinders with vertical ribs; metal caps are stamped crown corks with fluted crimps. Learnable. |
| **Broken Glass** | `Broken glass` | 193 | `glass` | **Consistent.** Sharp planar fracture edges, high transparency, and internal light refraction clearly separate broken glass from opaque litter. |
| **Cigarette Butts** | `Cigarette` | 1,128 | `general_trash` | **Highly Learnable.** Dense cluster of small cylindrical white/yellow filters with burnt tips. Highly distinct feature profile. |
| **Unlabeled Litter** | `Unlabeled litter` | 903 | `general_trash` | **Acceptable as Residual Class.** Consists of degraded debris that human annotators could not assign to a distinct material. Municipal sorting routes non-identifiable degraded litter directly to residual waste. |

### 3.2 Questionable Mappings Evaluated
* **`Aerosol` (17 instances):** Mapped to `metal`. Valid because empty cans are tinplate/aluminium. However, pressurized containers carry safety hazards. The web frontend must display safety warnings similar to batteries.
* **`Tissues` (75 instances):** Mapped to `paper`. Chemically cellulose paper. However, used napkins/tissues are non-recyclable due to sanitary/grease contamination. For *material detection*, `paper` is visually accurate; for *disposal guidance*, the system must advise general waste if soiled.

---

## 4. Checkpoint 3: Mathematical Count Reconciliation

All numbers across source raw data, remapped intermediate data, and final processed data reconcile perfectly:

```
Source Annotations (Raw):                                    8,152
  - Retained in Processed Dataset:                  8,119
  - Excluded Classes (Battery: 4, Food: 13,
                      Shoe: 11, Carded Blister: 2):    30
  - Degenerate Zero-Area Boxes Removed:                 3
Reconciliation Check: 8,119 + 30 + 3 = 8,152 (100.0% Reconciled)
```

### 4.1 Class ID Contiguity & YAML Alignment
* **`data/dataset.yaml` Configuration:**
  * `nc: 6`
  * `names:`
    * `0: plastic` (3,875 boxes)
    * `1: metal` (900 boxes)
    * `2: cardboard` (437 boxes)
    * `3: glass` (396 boxes)
    * `4: paper` (417 boxes)
    * `5: general_trash` (2,094 boxes)
* **Label Inspection:** Bounding boxes in `data/processed/labels/` contain integers strictly in $[0, 5]$. No orphaned or out-of-range class IDs exist.

### 4.2 Filesystem Audit vs. Manifest
* `data/processed/images/train/`: **1,790** images | `labels/train/`: **1,790** label files
* `data/processed/images/val/`: **379** images | `labels/val/`: **379** label files
* `data/processed/images/test/`: **374** images | `labels/test/`: **374** label files
* **Total:** **2,543 images and 2,543 labels** (Exact 1:1 match).

---

## 5. Checkpoint 4: Evaluation Readiness & Limitations

1. **Untouched Test Split:**
   * All 374 images and 1,106 bounding boxes in `data/processed/images/test/` are **SEALED**.
   * No training scripts or augmentations have touched this partition.
2. **Unsupported Classes Explicitly Documented:**
   * **`battery`:** **NOT SUPPORTED** in this model version (only 4 source instances existed). The UI will provide safety advisories but will not claim detection capabilities.
   * **`organic / food waste`:** **NOT SUPPORTED** in this model version (only 13 source instances existed).
3. **Open-Set Rejection Status:**
   * Standard YOLO models are closed-set. Confidence thresholding will be calibrated post-training using an independent negative/OOD evaluation set before deployment.

---

## 6. Automated Test Suite Results

```
Command: python -m pytest tests/test_data_preparation.py -v --tb=short
Platform: Windows (Python 3.14.6, pytest 9.2.0)

tests/test_data_preparation.py::TestParseLabel::test_valid_label                    PASSED
tests/test_data_preparation.py::TestParseLabel::test_invalid_class_id               PASSED
tests/test_data_preparation.py::TestParseLabel::test_coordinate_out_of_range        PASSED
tests/test_data_preparation.py::TestParseLabel::test_missing_fields                 PASSED
tests/test_data_preparation.py::TestParseLabel::test_empty_file                     PASSED
tests/test_data_preparation.py::TestParseLabel::test_missing_label_file             PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_valid_annotation_passes PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_degenerate_box_flagged  PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_skipped_for_missing_split PASSED
tests/test_data_preparation.py::TestDeduplication::test_md5_identical_files         PASSED
tests/test_data_preparation.py::TestDeduplication::test_md5_different_files         PASSED
tests/test_data_preparation.py::TestDeduplication::test_phash_same_image            PASSED
tests/test_data_preparation.py::TestDeduplication::test_phash_different_images      PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_get_primary_class          PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_split_ratios_sum            PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_full_audit_on_mock_dataset  PASSED
tests/test_data_preparation.py::TestLabelMapping::test_label_mapping_json_valid     PASSED
tests/test_data_preparation.py::TestLabelMapping::test_dataset_yaml_valid          PASSED
tests/test_data_preparation.py::TestLabelMapping::test_source_manifest_valid        PASSED

============================== 19 passed in 1.61s ==============================
```

---

## 7. Final Recommendation

**The dataset is verified and APPROVED for Step 3: Baseline Model Training (YOLOv11s).**
No further modifications to labels, splits, or dataset configurations are needed.

*Report logged in `docs/pretraining_quality_review.md`*

