# EcoVision AI — Label Fix & Polygon-to-BBox Conversion Review

**Document Version:** 1.0  
**Date:** October 29, 2026  
**Auditor / Author:** Senior ML Engineer & CV Architect  
**Workspace:** `D:\Projects\ml_project\472`  
**Target Dataset:** `data/processed_bbox_v2`  
**Reference Baseline Dataset:** `data/processed` (preserved intact as baseline evidence)  

---

## 1. Executive Summary

In accordance with [`docs/baseline_audit.md`](file:///d:/Projects/ml_project/472/docs/baseline_audit.md), the confirmed polygon-to-bounding-box truncation bug in the dataset preparation pipeline has been fully resolved. 

### Key Accomplishments
1. **Parser Overhaul:** [`scripts/label_mapper.py`](file:///d:/Projects/ml_project/472/scripts/label_mapper.py) was rewritten to correctly convert all $N$-vertex polygon segmentation masks (`class_id x1 y1 ... xn yn`) into exact, normalized YOLO object detection bounding boxes (`class_id cx cy w h`).
2. **Strict Quality Validation:** [`scripts/check_annotations.py`](file:///d:/Projects/ml_project/472/scripts/check_annotations.py) was enhanced to enforce exactly 5 tokens per detection label, strict bounds checking ($[0, 1] \pm 1e-4$), finite floats, and positive dimensions.
3. **Comprehensive Unit Testing:** A 15-test suite ([`tests/test_polygon_conversion.py`](file:///d:/Projects/ml_project/472/tests/test_polygon_conversion.py)) was created covering odd coordinate counts, boundary cases, minimal polygons, zero-area polygons, non-finite values, and taxonomy preservation. All **34 tests** across the project pass ($100\%$).
4. **Non-Destructive Regeneration:** The original `data/processed` directory remains completely untouched as historical evidence. The corrected dataset was written to **`data/processed_bbox_v2`**.
5. **Strict Split Preservation:** Using the manifest preservation mode in [`scripts/prepare_splits.py`](file:///d:/Projects/ml_project/472/scripts/prepare_splits.py), all 2,543 images were mapped to the exact same splits (`train`: 1,790, `val`: 379, `test`: 374). The sealed test split remains completely uncompromised.
6. **No Retraining Initiated:** Model retraining, checkpoint creation, and tuning remain on hold pending user review.

---

## 2. Summary of Files Changed & Added

| File Path | Action | Description |
| :--- | :---: | :--- |
| [`scripts/label_mapper.py`](file:///d:/Projects/ml_project/472/scripts/label_mapper.py) | **Modified** | Implemented `parse_annotation_to_bbox()`. Correctly computes $x_{\min}, x_{\max}, y_{\min}, y_{\max}$, centers $(cx, cy)$, and dimensions $(w, h)$ from all vertices. Validates coordinates within $[0, 1] \pm 1e-4$. Clamps float rounding artifacts. Rejects malformed polygons ($<6$ coordinates or odd counts). |
| [`scripts/check_annotations.py`](file:///d:/Projects/ml_project/472/scripts/check_annotations.py) | **Modified** | Enforces exact 5-token rule for detection. Validates finite numbers, non-zero positive dimensions, and verifies bounding box boundaries $x_1, y_1 \ge -1e-4$ and $x_2, y_2 \le 1.0 + 1e-4$. |
| [`scripts/prepare_splits.py`](file:///d:/Projects/ml_project/472/scripts/prepare_splits.py) | **Modified** | Added `--from_manifest` option to strictly preserve split assignments from `data/processed/split_manifest.json` during data generation. |
| [`tests/test_polygon_conversion.py`](file:///d:/Projects/ml_project/472/tests/test_polygon_conversion.py) | **Added** | 15 unit tests verifying polygon parsing, minimal triangles, degenerate shapes, odd coordinate counts, tolerance bounds, and 6-class mapping. |
| [`data/dataset_v2.yaml`](file:///d:/Projects/ml_project/472/data/dataset_v2.yaml) | **Added** | Dataset configuration pointing to `data/processed_bbox_v2` with the verified 6-class taxonomy. |
| `data/processed_bbox_v2/` | **Created** | Clean, corrected 6-class object detection dataset with true bounding boxes. |
| `evaluation/results_processed_v2/` | **Created** | Validation reports and comparative audit metrics for `processed_bbox_v2`. |

---

## 3. Unit Test Results

The test suite was executed via `pytest`:

```text
tests/test_data_preparation.py::TestParseLabel::test_valid_label PASSED
tests/test_data_preparation.py::TestParseLabel::test_invalid_class_id PASSED
tests/test_data_preparation.py::TestParseLabel::test_coordinate_out_of_range PASSED
tests/test_data_preparation.py::TestParseLabel::test_missing_fields PASSED
tests/test_data_preparation.py::TestParseLabel::test_empty_file PASSED
tests/test_data_preparation.py::TestParseLabel::test_missing_label_file PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_valid_annotation_passes PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_degenerate_box_flagged PASSED
tests/test_data_preparation.py::TestAnnotationChecker::test_skipped_for_missing_split PASSED
tests/test_data_preparation.py::TestDeduplication::test_md5_identical_files PASSED
tests/test_data_preparation.py::TestDeduplication::test_md5_different_files PASSED
tests/test_data_preparation.py::TestDeduplication::test_phash_same_image PASSED
tests/test_data_preparation.py::TestDeduplication::test_phash_different_images PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_get_primary_class PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_split_ratios_sum PASSED
tests/test_data_preparation.py::TestPrepareSplits::test_full_audit_on_mock_dataset PASSED
tests/test_data_preparation.py::TestLabelMapping::test_label_mapping_json_valid PASSED
tests/test_data_preparation.py::TestLabelMapping::test_dataset_yaml_valid PASSED
tests/test_data_preparation.py::TestLabelMapping::test_source_manifest_valid PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_valid_quadrilateral_polygon PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_minimal_triangle_polygon PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_odd_coordinate_count_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_fewer_than_3_points_rejected_as_polygon PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_non_finite_coordinates_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_out_of_bounds_coordinates_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_floating_point_tolerance_clamping PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_degenerate_zero_area_polygon_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_standard_5_token_bbox_passthrough PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_standard_bbox_out_of_bounds_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_negative_class_id_rejected PASSED
tests/test_polygon_conversion.py::TestPolygonToBBoxParsing::test_non_integer_class_id_rejected PASSED
tests/test_polygon_conversion.py::TestSixClassMappingPreservation::test_target_classes_exact PASSED
tests/test_polygon_conversion.py::TestSixClassMappingPreservation::test_excluded_classes_exact PASSED
tests/test_polygon_conversion.py::TestSixClassMappingPreservation::test_all_mappings_resolve_to_valid_targets PASSED

============================= 34 passed in 1.31s ==============================
```

---

## 4. Strict Annotation Quality Audit of `data/processed_bbox_v2`

The updated `scripts/check_annotations.py` was executed directly against `data/processed_bbox_v2`:

| Check / Metric | Train Split | Val Split | Test Split (Sealed) | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Image Count** | 1,790 | 379 | 374 | **100% Match** |
| **Label Count** | 1,790 | 379 | 374 | **100% Match** |
| **Field Count Errors (`!= 5`)** | 0 | 0 | 0 | **0 Errors** |
| **Invalid Class IDs (`< 0` or `> 5`)** | 0 | 0 | 0 | **0 Errors** |
| **Non-Finite Values (`NaN`/`Inf`)** | 0 | 0 | 0 | **0 Errors** |
| **Degenerate Boxes ($w \le 0$ or $h \le 0$)** | 0 | 0 | 0 | **0 Errors** |
| **Out-of-Bounds Errors ($> 1e-4$)** | 0 | 0 | 0 | **0 Errors** |
| **Large Box Anomalies ($> 95\%$ area)** | 0 | 0 | 0 | **0 Errors** |
| **Orphan Images or Labels** | 0 | 0 | 0 | **0 Errors** |
| **Valid Annotations Scanned** | **5,939** | **1,077** | **1,106** | **8,122 Total** |

---

## 5. Comparative Statistics: Old Baseline (`v1`) vs Corrected (`v2`)

### A. Split & Annotation Counts
- **Total Images:** 2,543 in both versions. Split assignment is **100% identical** across `train` (1,790), `val` (379), and `test` (374).
- **Total Annotations:**
  - **Old (`v1`):** 8,119 boxes (3 boxes were mistakenly dropped as "degenerate" because vertex 2 coordinate was 0.0).
  - **New (`v2`):** **8,122 boxes** (+3 recovered: 2 in `plastic`, 1 in `cardboard`).
  - **Reconciliation:** All 8,152 raw annotations are accounted for: $8,122 \text{ mapped} + 30 \text{ excluded (null)} = 8,152 \text{ total}$.

### B. Class Distribution Comparison Across Splits
| Class ID | Target Class | Train (v1 $\rightarrow$ v2) | Val (v1 $\rightarrow$ v2) | Test (v1 $\rightarrow$ v2) | Total (v1 $\rightarrow$ v2) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | `plastic` | $2,816 \rightarrow \mathbf{2,817}$ | $536 \rightarrow \mathbf{537}$ | $523 \rightarrow \mathbf{523}$ | $3,875 \rightarrow \mathbf{3,877}$ |
| `1` | `metal` | $643 \rightarrow \mathbf{643}$ | $124 \rightarrow \mathbf{124}$ | $133 \rightarrow \mathbf{133}$ | $900 \rightarrow \mathbf{900}$ |
| `2` | `cardboard` | $309 \rightarrow \mathbf{310}$ | $59 \rightarrow \mathbf{59}$ | $69 \rightarrow \mathbf{69}$ | $437 \rightarrow \mathbf{438}$ |
| `3` | `glass` | $304 \rightarrow \mathbf{304}$ | $53 \rightarrow \mathbf{53}$ | $39 \rightarrow \mathbf{39}$ | $396 \rightarrow \mathbf{396}$ |
| `4` | `paper` | $296 \rightarrow \mathbf{296}$ | $54 \rightarrow \mathbf{54}$ | $67 \rightarrow \mathbf{67}$ | $417 \rightarrow \mathbf{417}$ |
| `5` | `general_trash` | $1,569 \rightarrow \mathbf{1,569}$ | $250 \rightarrow \mathbf{250}$ | $275 \rightarrow \mathbf{275}$ | $2,094 \rightarrow \mathbf{2,094}$ |
| **Total** | — | **$5,937 \rightarrow \mathbf{5,939}$** | **$1,076 \rightarrow \mathbf{1,077}$** | **$1,106 \rightarrow \mathbf{1,106}$** | **$8,119 \rightarrow \mathbf{8,122}$** |

### C. Bounding Box Area & Size Distribution Comparison
The true object geometry has been restored:

| Metric | Train Split (v1 vs v2) | Val Split (v1 vs v2) | Test Split (v1 vs v2) | Overall Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Small (<32×32 px)** | $0.7\% \rightarrow \mathbf{46.4\%}$ | $0.3\% \rightarrow \mathbf{41.0\%}$ | $0.6\% \rightarrow \mathbf{42.2\%}$ | **Restored true small litter** |
| **Medium (32–96 px)** | $3.4\% \rightarrow \mathbf{32.6\%}$ | $3.1\% \rightarrow \mathbf{36.1\%}$ | $2.3\% \rightarrow \mathbf{34.2\%}$ | **Normal scale representation** |
| **Large (>96×96 px)** | $96.0\% \rightarrow \mathbf{21.0\%}$ | $96.7\% \rightarrow \mathbf{22.8\%}$ | $97.1\% \rightarrow \mathbf{23.6\%}$ | **Eliminated giant false boxes** |
| **Median Area (Norm.)** | $0.2327 \rightarrow \mathbf{0.0032}$ | $0.2304 \rightarrow \mathbf{0.0050}$ | $0.2374 \rightarrow \mathbf{0.0041}$ | **46x to 72x Area Reduction** |
| **Maximum Area** | $0.9516 \rightarrow \mathbf{0.7240}$ | $0.8383 \rightarrow \mathbf{0.9423}$ | $0.9992 \rightarrow \mathbf{0.7684}$ | **Canvas overflow eliminated** |

---

## 6. Sealed Test Split Integrity

The test partition in `data/processed_bbox_v2/images/test/` and `data/processed_bbox_v2/labels/test/`:
- Contains exactly the **374 images** and **1,106 annotations** originally assigned.
- Was generated purely by applying the deterministic polygon conversion to the source annotations.
- Has **not** been used for evaluation, threshold tuning, epoch selection, or hyperparameter searching.
- Remains strictly **sealed** until Step 4 independent benchmark evaluation.

---

## 7. Remaining Considerations for Retraining

1. **Small-Object Sensitivity:** Now that true object dimensions are restored, $45.0\%$ of objects are $<32\times 32$ pixels. In standard YOLO training at $640\times 640$, mosaic augmentation (`mosaic: 1.0`) stitches 4 images together, shrinking some objects below $16\times 16$ pixels. During Step 3 retraining, consider testing `mosaic: 0.5` or reducing mosaic epochs in the final 10 epochs (standard Ultralytics `close_mosaic=10`).
2. **Class Imbalance:** `plastic` ($3,877$) and `general_trash` ($2,094$) remain the majority classes, while `cardboard` ($438$), `glass` ($396$), and `paper` ($417$) are less frequent. This is a property of the TACO dataset distribution, but the model will no longer be sabotaged by corrupt spatial coordinates.

---

## 8. Google Colab Packaging & Archive Verification

The corrected dataset has been packaged into a clean, standalone ZIP archive ready for Google Drive / Google Colab upload:

- **Archive File Name:** `ecovision_bbox_v2.zip`
- **Primary Path:** [`data/ecovision_bbox_v2.zip`](file:///d:/Projects/ml_project/472/data/ecovision_bbox_v2.zip)
- **Root Path:** [`ecovision_bbox_v2.zip`](file:///d:/Projects/ml_project/472/ecovision_bbox_v2.zip)
- **Archive Size:** `270,349,641 bytes` (**257.83 MB**)
- **SHA-256 Checksum:** `eb140c00ed14e968245f43f8339f515aca78ab18853dbe1e26407ff0b9a4e525`
- **Total Archived Entries:** 5,092 files

### Archive Verification Checklist
- [x] **Images & Labels:** Exactly 1,790 train, 379 val, and 374 test image-label pairs.
- [x] **Colab Configuration YAML:** `dataset_colab.yaml` and `data/dataset_colab_v2.yaml` pointing to `/content/data/processed_bbox_v2`.
- [x] **Portable Configuration YAML:** `dataset.yaml` and `data/dataset_v2.yaml`.
- [x] **Metadata Manifests:** `data/processed_bbox_v2/split_manifest.json` and `data/processed_bbox_v2/class_distribution.json`.
- [x] **Zero Contamination:** 0 files from old `data/processed/` and 0 `.pt` model weight files.


