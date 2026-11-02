# EcoVision AI — Step 1: YOLO11m Colab Training Preparation Report

**Document:** `docs/yolo11m_training_prep_report.md`  
**Date:** October 30, 2026  
**Status:** Step 1 Completed (Awaiting User Approval Before Execution)  
**Target Notebook:** [`notebooks/ecovision_yolo11m_training.ipynb`](file:///D:/Projects/ml_project/472/notebooks/ecovision_yolo11m_training.ipynb)  
**Candidate Checkpoint Destination:** `models/experiments/yolo11m_best.pt` (Zero modification to `models/weights_best.pt`)

---

## 1. Files Created & Modified

| File | Status | Description |
|---|---|---|
| [`notebooks/ecovision_yolo11m_training.ipynb`](file:///D:/Projects/ml_project/472/notebooks/ecovision_yolo11m_training.ipynb) | **Created** | Self-contained, reproducible 29-cell Google Colab notebook covering GPU verification through training, fair baseline comparison, dense mixed-waste benchmarking, and checkpoint export. |
| [`scripts/build_yolo11m_notebook.py`](file:///D:/Projects/ml_project/472/scripts/build_yolo11m_notebook.py) | **Created** | Automated notebook generator script with strict schema validation. |
| [`docs/yolo11m_training_prep_report.md`](file:///D:/Projects/ml_project/472/docs/yolo11m_training_prep_report.md) | **Created** | This preparation report. |

*Note: The dataset archive `data/ecovision_v3.zip` was verified to be 100% complete; no additional packaging scripts or dataset file modifications were required.*

---

## 2. Dataset Configuration Verification Results

A complete integrity and path audit was conducted on `data/ecovision_v3.zip` and `data/dataset_colab_v3.yaml`:

### Archive & Split Integrity
- **Archive File:** `data/ecovision_v3.zip` (656,080,034 bytes / 625.69 MB)
- **Verified SHA-256 Checksum:** `0c7feba5af5e340e44ce8ae2a222b992bab4c5185f7ec43cb3aaa5ad05543627`
- **Total Files in Archive:** 23,250 files
- **Image & Label Split Counts:**
  - **Train:** 8,242 images | 8,242 labels
  - **Val:** 3,007 images | 3,007 labels
  - **Test (Sealed):** 374 images | 374 labels
  - **Total:** 11,623 images | 11,623 labels

### Taxonomy Verification (6 Classes)
All annotations strictly conform to the approved 6-class taxonomy:
- `0`: `plastic`
- `1`: `metal`
- `2`: `cardboard`
- `3`: `glass`
- `4`: `paper`
- `5`: `general_trash`

### Colab YAML Compatibility
`data/dataset_colab_v3.yaml` configures:
```yaml
path: /content/dataset/data/processed_v3
train: images/train
val: images/val
test: images/test
names:
  0: plastic
  1: metal
  2: cardboard
  3: glass
  4: paper
  5: general_trash
nc: 6
```
When extracted with `!unzip -q ecovision_v3.zip -d /content/dataset`, the directories match `path` exactly. In addition, Section 5 of the notebook programmatically validates on-disk directories and writes a canonical absolute YAML to `/content/dataset_colab_v3.yaml` to prevent any runtime directory mismatch.

---

## 3. Augmentation Investigation Findings (Task 3)

We investigated the Ultralytics 8.4 engine internals (`ultralytics.data.augment`) to verify whether Mosaic, MixUp, and Copy-Paste are suitable:

1. **`mosaic: 1.0` (Supported & Recommended):**  
   - Ultralytics natively stitches 4 random training images into a single composite mosaic, concatenating all bounding boxes.  
   - This directly counters the dataset's 48.4% single-object bias by synthesizing multi-object mixed scenes during 85% of training.
2. **`mixup: 0.15` (Supported & Recommended):**  
   - Linearly blends two images and combines their bounding boxes.  
   - Prevents the network from overfitting to specific backgrounds or clean lab environments.
3. **`copy_paste: 0.0` (Unsupported for Bounding Boxes — Critical Finding):**  
   - Inspection of `ultralytics/data/augment.py` (`CopyPaste.__call__`) shows:
     ```python
     if len(labels["instances"].segments) == 0 or self.p == 0:
         return labels
     ```
   - In Ultralytics, `CopyPaste` **requires segmentation polygon masks**.  
   - Because 77.8% of annotations in `data/processed_v3` (28,539 annotations from General Waste Data) are 5-token bounding boxes without polygon segmentation masks, `CopyPaste` is either bypassed or introduces rectangular crop artifacts.  
   - **Decision:** Set `copy_paste: 0.0` and rely on `mosaic: 1.0` + `mixup: 0.15`.
4. **`close_mosaic: 15` (Supported & Recommended):**  
   - Mosaic is disabled during the final 15 epochs, allowing the detection head to optimize precise bounding-box coordinates on natural uncropped imagery.

---

## 4. Notebook Section Overview (14 Sections)

The notebook [`notebooks/ecovision_yolo11m_training.ipynb`](file:///D:/Projects/ml_project/472/notebooks/ecovision_yolo11m_training.ipynb) is organized into 14 distinct, structured modules:

| Section | Title | Primary Function |
|---|---|---|
| **1** | **Google Colab GPU Setup** | Inspects `nvidia-smi`, queries PyTorch CUDA runtime, checks total VRAM, and raises a clear error if GPU is not assigned. |
| **2** | **Dependency Installation** | Installs pinned `ultralytics>=8.3.0,<8.5.0`, `pyyaml`, `pandas`, `tabulate`, and `matplotlib`. |
| **3** | **Dataset Upload / Drive Mount** | Dual-mode: automatically locates `ecovision_v3.zip` in Google Drive or accepts direct file upload, then validates SHA-256 checksum. |
| **4** | **Dataset Extraction** | Extracts archive into Colab NVMe storage (`/content/dataset/`) and validates file counts in all splits. |
| **5** | **Dataset YAML Verification** | Validates filesystem paths and generates a canonical runtime YAML. |
| **6** | **Class ID & Annotation Validation** | Verifies normalized coordinates ($0 \le x, y, w, h \le 1$) and class ID ranges ($0 \dots 5$). |
| **7** | **Existing YOLO11s Baseline Evaluation** | Loads existing `models/weights_best.pt` and evaluates it on the exact same validation split (`conf=0.15`, `imgsz=640`) to establish ground truth comparison. |
| **8** | **YOLO11m Configuration** | Dynamically probes GPU VRAM to select safe batch size (`16` if VRAM $\ge 15$ GB, `8` with `accumulate=2` if VRAM is constrained). |
| **9** | **Model Training Execution** | Executes `model.train()` with automatic CUDA OOM recovery fallback. |
| **10** | **Best Checkpoint Verification** | Verifies `best.pt` existence, size (~40 MB), and SHA-256 hash. |
| **11** | **Validation Metrics & Comparison** | Runs `model.val()` on YOLO11m and outputs a side-by-side Markdown comparison table (Overall + Per-Class). |
| **12** | **Dense Mixed-Waste Evaluation** | Evaluates and plots predictions on dense mixed-waste validation scenes ($\ge 4$ objects, $\ge 2$ classes). |
| **13** | **Size & Latency Benchmarks** | Measures GPU latency (ms), simulated CPU latency (ms), model size, and parameters. |
| **14** | **Checkpoint Export** | Packages `yolo11m_best.pt`, `results.csv`, `metrics_summary.json`, and confusion matrices into `yolo11m_candidate_package.zip`, backing up to Google Drive and triggering browser download. |

---

## 5. Expected Google Colab GPU Requirements

| Parameter | Minimum Requirement | Recommended (Standard Free Tier) |
|---|---|---|
| **GPU Model** | NVIDIA Tesla T4 (15.3 GB VRAM) | NVIDIA Tesla T4 or L4 / A100 |
| **GPU Memory Usage** | ~8.2 GB (at `batch=8`) | ~11.8 GB (at `batch=16`, `imgsz=960`) |
| **Colab System RAM** | 12.7 GB (Standard) | 12.7 GB |
| **Disk Space Needed** | ~15 GB (Archive + Extracted images + Run artifacts) | ~25 GB (Free tier provides ~78 GB) |
| **Estimated Training Duration** | ~3.5 to 4.0 hours (Tesla T4, 100 epochs with early stopping) | ~3.0 hours (if early stopping triggers near epoch 60-70) |

---

## 6. How to Run the Training Notebook

1. **Upload Dataset Archive to Google Drive:**
   - On local machine, copy `D:\Projects\ml_project\472\data\ecovision_v3.zip`.
   - In Google Drive, create a folder named `EcoVisionAI` and upload `ecovision_v3.zip`.
   - *(Optional for Baseline Comparison)* Also upload `models/weights_best.pt` to `MyDrive/EcoVisionAI/weights_best.pt` so Section 7 can run the side-by-side comparison.
2. **Open Notebook in Google Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com).
   - Click **Upload** and select `notebooks/ecovision_yolo11m_training.ipynb`.
3. **Set Hardware Accelerator:**
   - Go to **Runtime** $\rightarrow$ **Change runtime type**.
   - Hardware accelerator: **T4 GPU**.
   - Click **Save**.
4. **Execute Cells Step-by-Step:**
   - Run Cells 1 through 6 to mount Drive, extract the data, and verify annotations.
   - Run Cell 7 to benchmark the baseline YOLO11s.
   - Run Cells 8 and 9 to launch YOLO11m training.
   - Run Cells 10 through 14 to generate validation metrics, dense scene plots, latency comparisons, and export the package.
5. **Retrieve Candidate Checkpoint:**
   - Download `yolo11m_candidate_package.zip`.
   - Place extracted candidate weights into `models/experiments/yolo11m_best.pt` locally.

---

## 7. Safeguards & Unresolved Issues

### Strict Safeguards Confirmed
- `models/weights_best.pt` is **untouched**.
- `data/real_world_eval/` remains **sealed, unopened, and unreferenced**.
- The existing running FastAPI backend and React frontend are **unmodified and operational**.
- Training will **NOT** run automatically.

### Unresolved Issues / Prerequisites for the User
1. **Google Drive Upload:** The user must upload `data/ecovision_v3.zip` (625 MB) to their Google Drive before executing the notebook.
2. **Free Colab Session Time:** A standard Colab free session allows continuous execution for up to 12 hours (with active browser tab). The expected training time of ~3.5 hours fits comfortably within this window, but the user should ensure the tab remains open during training.

