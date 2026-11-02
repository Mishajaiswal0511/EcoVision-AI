# EcoVision AI — Pre-Training Checklist & Limitations Audit (v3)

**Document Version:** 1.0  
**Date:** October 29, 2026  
**Auditor / Lead Engineer:** Senior Computer Vision Engineer & ML Architect  
**Project Workspace:** `D:\Projects\ml_project\472`  
**Dataset:** `data/processed_v3/` (11,623 images, 36,661 boxes)  
**Target Model:** Ultralytics YOLO11s (`yolo11s.pt`)  
**Target Platform:** Google Colab GPU (NVIDIA T4 / V100 / A100)  

---

## 1. Pre-Training Readiness Verification Checklist

| Item | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **1. Target Taxonomy** | Strict 6 classes: `0: plastic`, `1: metal`, `2: cardboard`, `3: glass`, `4: paper`, `5: general_trash`. | Audited via YAML, test suite, and split manifest. | **PASS** |
| **2. Bounding Box Integrity** | All labels must have exactly 5 tokens, finite values, and boundaries in $[0, 1] \pm 10^{-4}$. | `scripts/check_annotations.py` checked all 36,661 boxes. 0 errors. | **PASS** |
| **3. Negative Supervision** | 800 certified indoor background images with empty (0-byte) `.txt` labels. | Verified with COCO ground truth; 0 prohibited objects present. | **PASS** |
| **4. Sealed Test Independence** | Exactly 374 TACO test images with 1,106 boxes; 0 duplicate or near-duplicate leaks into train/val. | Pairwise pHash & MD5 verified with Hamming distance $> 4$. | **PASS** |
| **5. Sealed Real-World Benchmark** | 11 out-of-distribution benchmark scenes kept isolated in `data/real_world_eval/`. | Verified unreferenced by any training, validation, or tuning script. | **PASS** |
| **6. Hyperparameter Compatibility** | Confirm `fl_gamma` deprecation in Ultralytics YOLO11 and ensure supported arguments. | Verified Ultralytics `cfg/default.yaml`; legacy `fl_gamma` removed. | **PASS** |
| **7. False Positive Controls** | `mosaic: 0.5`, `close_mosaic: 10`, `mixup: 0.05` configured. | Embedded in `training/configs/yolov11_waste_v3.yaml` and Colab notebook. | **PASS** |
| **8. Colab Compatibility** | Relative / `/content/dataset` paths in YAML; reproducible notebook created. | Tested in `training/colab_yolov11_v3_training.ipynb`. | **PASS** |
| **9. Packaging & Checksum** | Archive generated with SHA-256 hash and sandbox extraction verification. | Verified `data/ecovision_v3.zip` (`0c7feba5af5e340e44ce8ae2a222b992bab4c5185f7ec43cb3aaa5ad05543627`). | **PASS** |
| **10. Training Restraint** | Zero model training executed before explicit user sign-off. | Training strictly withheld. | **PASS** |

---

## 2. Technical Limitations & Real-World Operating Boundaries

While the v3 dataset represents a monumental improvement over baseline v1/v2 (expanding from 2,543 images to 11,623 images and introducing native negative supervision), the following real-world boundaries must be acknowledged:

### 1. Intentional Organic / Biodegradable Omission
- **Boundary:** To avoid skewing the recyclable detection head with 45,395 food/leaf annotations, all `BIODEGRADABLE` items were strictly excluded.
- **Consequence:** The model **cannot** detect organic waste (e.g. apple cores, banana peels, vegetable trimmings, compost). If placed on a sorting conveyor, food items will be ignored as background unless contaminated with recyclable packaging.

### 2. High-Entropy General Trash
- **Boundary:** In both TACO and municipal waste streams, `general_trash` consists of irregular, torn, dirty, and non-recyclable composite materials (cigarette butts, candy wrappers, fragmented plastic films).
- **Consequence:** While precision on rigid recyclables (`plastic`, `metal`, `glass`, `cardboard`) is expected to increase substantially, `general_trash` inherently retains higher variance due to morphological heterogeneity.

### 3. Notebooks and Paper/Cardboard Edge Cases
- **Boundary:** Physical spiral and bound notebooks contain stiff paperboard covers (physically cardboard) encasing bleached sheets (physically paper).
- **Consequence:** In unconstrained desktop scenes, notebooks remain a dual-material object. While negative images prevent random hallucination on empty desks, a visible open notebook may produce dual detections unless explicitly post-processed.

### 4. Severe Crushing and Soil Soiling
- **Boundary:** TACO images feature outdoor roadside/beach litter, while General Waste Data features municipal sorting facility items.
- **Consequence:** Aluminum cans or plastic bottles that are crushed flat or covered in mud may produce lower confidence scores than clean, intact packaging.

### 5. Single GPU Memory Constraints
- **Boundary:** Batch size is set to 16 at $640 \times 640$. On standard Google Colab T4 GPUs (15.8 GB VRAM), memory usage is expected to peak around 6–8 GB. If multi-scale training or larger batch sizes are desired, Google Colab Pro (A100) or reduction to `batch: 8` may be required.

---

## 3. Pre-Training Sign-Off Summary

Phase 2 preparations are structured so that execution in Google Colab is completely self-contained, reproducible, and verifiable. No metrics have been fabricated or presumed. Model training remains on hold pending your final approval.

