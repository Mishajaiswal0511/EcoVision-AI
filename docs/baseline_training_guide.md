# EcoVision AI — Baseline Model Training Guide (Google Colab)

**Document Version:** 1.0  
**Date:** October 29, 2026  
**Target Architecture:** Ultralytics YOLO11s (`yolo11s.pt`)  
**Task:** 6-Class Real-World Waste Object Detection  

---

## 1. Executive Summary & Training Objectives

EcoVision AI aims to provide robust waste segregation guidance from real-world smartphone imagery. Following rigorous dataset audits, semantic remapping, deduplication, and pre-training validation, model training is conducted on **Google Colab GPU** (NVIDIA T4 or better).

### Core Training Principles
- **Honest Baseline:** We establish an authentic, reproducible baseline without synthetic metrics or exaggerated claims.
- **Sealed Test Split:** The test split (`374` images, `1,106` bounding boxes) is strictly **sealed**. Validation during training is conducted exclusively against the validation split (`379` images, `1,228` bounding boxes).
- **Free Cloud Compute:** Training executes entirely within Google Colab's free GPU tier; no heavy compute or CUDA runtime is required on the local development machine.

---

## 2. Model & Checkpoint Specifications

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Model Architecture** | `YOLO11s` (Small) | Balanced compute footprint (~9.4M parameters, 21.5 GFLOPs), ideal for real-time edge/browser inference with strong feature representation. |
| **Pretrained Weights** | `yolo11s.pt` (COCO-pretrained) | Ultralytics official weights pretrained on 80 COCO classes, providing rich low-level edge, texture, and object priors. |
| **Framework** | Ultralytics 8.3+ / PyTorch 2.0+ | Official PyTorch implementation of YOLO11. |
| **License** | AGPL-3.0 (Ultralytics) | Open-source research and educational usage. |
| **Input Resolution** | $640 \times 640$ pixels | Standard resolution preserving fine detail (bottle caps, litter) while fitting Colab 16 GB VRAM. |
| **Batch Size** | 16 (Fallback: 8) | Stable batch norm statistics; fits standard T4 16GB VRAM at 640px. |
| **Target Epochs** | 100 epochs (Patience = 20) | Full convergence envelope with early stopping based on `val/mAP50-95`. |
| **Optimizer** | `AdamW` ($\text{lr}_0 = 0.001$, $\beta_1 = 0.937$, $\text{decay} = 0.0005$) | Proven stability for small datasets and fine-tuning transfer learning. |
| **Reproducibility Seed** | `42` | Ensures deterministic data shuffling and parameter initialization. |

---

## 3. Dataset & Taxonomy Verification

The model is trained on the verified 6-class EcoVision taxonomy:

| Class ID | Class Name | Train Instances | Val Instances | Municipal Segregation Action |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `plastic` | 2,752 | 586 | Dry recycling stream (Blue bin) |
| `1` | `metal` | 647 | 134 | Dry recycling stream (Blue bin) |
| `2` | `cardboard` | 315 | 63 | Dry recycling stream (Flattened) |
| `3` | `glass` | 288 | 56 | Dedicated glass stream / safe handling |
| `4` | `paper` | 303 | 55 | Dry recycling stream (Clean & dry) |
| `5` | `general_trash` | 1,480 | 334 | Residual non-recyclable municipal waste |
| **Total** | **6 classes** | **5,785** | **1,228** | — |

> [!IMPORTANT]
> The test set (`374` images, `1,106` annotations) is **NEVER** evaluated during training epochs or used for early stopping. It remains untouched for the Step 4 independent benchmark evaluation.

---

## 4. Google Colab Step-by-Step Execution Guide

### Step 4.1: Prepare the Dataset Archive
On your local machine, run the packager script to build the Colab archive:
```powershell
python scripts/package_colab_dataset.py
```
This produces `data/ecovision_processed_dataset.zip` (~200 MB), containing `data/processed/` and `data/dataset.yaml`.

### Step 4.2: Upload to Google Drive (Recommended)
1. Open [Google Drive](https://drive.google.com).
2. Create a folder named: `EcoVisionAI`.
3. Upload `ecovision_processed_dataset.zip` directly into `MyDrive/EcoVisionAI/`.
4. Also upload the notebook [`training/colab_baseline.ipynb`](file:///d:/Projects/ml_project/472/training/colab_baseline.ipynb) or open it directly in Google Colab.

### Step 4.3: Open and Configure Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `training/colab_baseline.ipynb`.
3. In the top navigation bar, select **Runtime** $\rightarrow$ **Change runtime type**.
4. Set **Hardware accelerator** to **T4 GPU** (or any available GPU).
5. Click **Save**.

### Step 4.4: Execute the Notebook Cells

The notebook is divided into clear, numbered sections:

1. **Cell 1 — GPU Verification:**
   - Runs `!nvidia-smi` and checks CUDA driver availability.
   - Confirms PyTorch detects the GPU (`torch.cuda.get_device_name(0)`).
2. **Cell 2 — Install Dependencies:**
   - Installs `ultralytics` and supporting scientific packages cleanly.
3. **Cell 3 & 4 — Mount Drive & Extract Dataset:**
   - Mounts Google Drive (`drive.mount('/content/drive')`).
   - Copies and extracts `ecovision_processed_dataset.zip` into `/content/data/processed/`.
   - Validates that image and label files exist in `train`, `val`, and `test`.
4. **Cell 5 — Colab Dataset YAML:**
   - Generates `/content/dataset_colab.yaml` pointing directly to `/content/data/processed`.
5. **Cell 6 — Initialize YOLO11s Checkpoint:**
   - Downloads official `yolo11s.pt` and displays parameter counts.
6. **Cell 7 — Execute Training:**
   - Launches `model.train()` with parameters:
     - `epochs=100`, `imgsz=640`, `batch=16`, `seed=42`, `patience=20`.
   - Automatically handles CUDA Out-of-Memory (OOM) errors by catching exceptions and retrying with `batch=8`.
7. **Cell 8 — Validation & Metric Summary:**
   - Runs validation on `val` split.
   - Extracts and prints per-class mAP@50 and mAP@50-95 in an easy-to-read table.
8. **Cell 9 — Export and Save Artifacts to Drive:**
   - Copies `best.pt`, `last.pt`, `results.csv`, `confusion_matrix.png`, and `val_batch0_labels.jpg` to `/content/drive/MyDrive/EcoVisionAI/runs/`.

---

## 5. Expected Output & Artifact Structure

When training concludes, Colab will have generated the following artifacts under `/content/drive/MyDrive/EcoVisionAI/runs/baseline_yolo11s/`:

```
baseline_yolo11s/
|-- weights/
|   |-- best.pt                     <- Checkpoint with highest validation mAP50-95
|   |-- last.pt                     <- Final epoch checkpoint (for resumption)
|-- confusion_matrix.png            <- Unnormalized confusion matrix
|-- confusion_matrix_normalized.png <- Normalized confusion matrix (shows class confusion)
|-- F1_curve.png                    <- F1 score vs confidence threshold curves
|-- PR_curve.png                    <- Precision-Recall curves per class
|-- P_curve.png                     <- Precision vs confidence
|-- R_curve.png                     <- Recall vs confidence
|-- results.csv                     <- Per-epoch train/val loss and mAP telemetry
|-- results.png                     <- Composite plots of all training metrics over time
|-- args.yaml                       <- Exact hyperparameter configuration used
```

---

## 6. viva / Defense Q&A: Key Technical Decisions

> **Q: Why use YOLO11s instead of YOLO11x or a larger model?**  
> **A:** YOLO11s has ~9.4M parameters, offering an optimal trade-off between feature expressiveness and inference latency. Waste segregation applications often deploy on edge devices (smart bins, raspberry pi, mobile phones) where larger models (>50M params) suffer prohibitive latency and memory overhead without proportional accuracy gains on moderate dataset sizes.

> **Q: Why was the test split completely omitted from training and tuning?**  
> **A:** Using the test split to tune hyperparameters or select the `best.pt` checkpoint introduces data snooping and optimism bias. To report true generalization performance in research and viva evaluation, the test split must remain pristine until final benchmarking.

> **Q: How does the training pipeline prevent GPU Out-of-Memory (OOM) crashes?**  
> **A:** Colab free T4 GPUs offer 15-16 GB of VRAM. With batch size 16 and image size 640, peak VRAM is ~7.5 GB. If VRAM spikes occur, the notebook wraps training in a try-except block that flushes CUDA cache and restarts training with batch size 8.

