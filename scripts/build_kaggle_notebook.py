import json
from pathlib import Path

def create_kaggle_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_markdown(r"""# EcoVision AI — Headless Kaggle GPU Training & Baseline Benchmark
**Project:** EcoVision AI (`D:\Projects\ml_project\472`)  
**Target Model:** Ultralytics YOLO11m Object Detection (`yolo11m.pt`)  
**Input Resolution:** $960 \times 960$ pixels  
**Hardware Accelerator:** Kaggle Cloud GPU (NVIDIA Tesla T4 / P100 16GB VRAM)  
**Taxonomy (6 Classes):** `0: plastic`, `1: metal`, `2: cardboard`, `3: glass`, `4: paper`, `5: general_trash`  
**Dataset Split Scope:** Train (8,242 images) & Validation (3,007 images) ONLY. (Sealed test split strictly excluded).  
**Execution Mode:** Headless CLI managed by Antigravity  

---
### Workflow Pipeline
1. GPU Hardware & CUDA Verification.
2. Dependency Installation (`ultralytics>=8.3.0,<8.5.0`, `pyyaml`, `pandas`, `tabulate`).
3. Dataset Mount / Extraction from `/kaggle/input/` to `/tmp/dataset/`.
4. Runtime Dataset YAML path configuration & annotation integrity check.
5. Benchmark Baseline YOLO11s (`conf=0.15`, `imgsz=640`).
6. Train YOLO11m (`imgsz=960`, `epochs=100`, `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.0`, `patience=25`, safe fallback: 16 -> 8 -> 4).
7. Full Validation & Side-by-Side Comparison (Precision, Recall, mAP50, mAP50-95, per-class metrics).
8. Dense Mixed-Waste Scene Evaluation ($\ge 4$ objects, $\ge 2$ classes).
9. Export Best Candidate Checkpoint to `/kaggle/working/yolo11m_best.pt` for automated CLI pull.""")

    # Section 1
    add_markdown("""## Section 1: Verify Kaggle GPU Accelerator""")
    add_code("""!nvidia-smi

import torch
import sys

print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CRITICAL: CUDA GPU is not available! Kaggle kernel must have GPU enabled.")

gpu_name = torch.cuda.get_device_name(0)
gpu_props = torch.cuda.get_device_properties(0)
total_vram_gb = gpu_props.total_memory / (1024 ** 3)
print(f"Assigned GPU   : {gpu_name}")
print(f"Total VRAM     : {total_vram_gb:.2f} GB")
""")

    # Section 2
    add_markdown("""## Section 2: Install Ultralytics & Core Dependencies""")
    add_code("""!pip install --quiet --upgrade "ultralytics>=8.3.0,<8.5.0" pyyaml pandas tabulate matplotlib seaborn

import ultralytics
import yaml
import pandas as pd
import numpy as np

print(f"Ultralytics version: {ultralytics.__version__}")
""")

    # Section 3
    add_markdown("""## Section 3: Dataset Staging from Kaggle Input to Scratch Space
Detect uncompressed or compressed dataset in `/kaggle/input/` and stage in `/tmp/dataset/data/processed_v3`.""")
    add_code("""import os
import zipfile
import shutil
from pathlib import Path

input_dir = Path('/kaggle/input')
print(f"Inspecting input directory: {input_dir}")
for p in input_dir.iterdir():
    print(f"  Available input source: {p.name}")

# Check for uncompressed processed_v3 first (standard Kaggle unpack)
processed_dirs = list(input_dir.glob('**/processed_v3'))
dataset_root = None

if processed_dirs:
    src_root = processed_dirs[0]
    print(f"Found uncompressed dataset at: {src_root}")
    target_dir = Path('/tmp/dataset/data/processed_v3')
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if not target_dir.exists():
        print(f"Copying {src_root} -> {target_dir} for fast local NVMe access...")
        shutil.copytree(src_root, target_dir)
    dataset_root = target_dir
else:
    # Check for zip archives
    zip_candidates = list(input_dir.glob('**/*.zip'))
    if zip_candidates:
        zip_path = zip_candidates[0]
        print(f"Found dataset archive: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
        extract_dir = Path('/tmp/dataset')
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        dataset_root = extract_dir / 'data' / 'processed_v3' if (extract_dir / 'data' / 'processed_v3').exists() else extract_dir
    else:
        raise FileNotFoundError(f"Could not find processed_v3 or zip in {input_dir}!")

print(f"Verified dataset root: {dataset_root}")
for split in ['train', 'val']:
    n_img = len(list((dataset_root / 'images' / split).glob('*.*')))
    n_lbl = len(list((dataset_root / 'labels' / split).glob('*.txt')))
    print(f"Split [{split:5s}]: {n_img:5d} images | {n_lbl:5d} labels")

# Security assertion: ensure test split is completely absent
test_dir = dataset_root / 'images' / 'test'
assert not test_dir.exists() or len(list(test_dir.glob('*.*'))) == 0, "SECURITY ERROR: Test data detected in training environment!"
print("Security check passed: Test split is strictly absent.")
""")

    # Section 4
    add_markdown("""## Section 4: Dataset YAML Runtime Configuration
Generate absolute YAML configuration pointing to `/tmp/dataset/data/processed_v3` (Train and Validation ONLY).""")
    add_code("""DATASET_YAML = '/tmp/dataset_train_val.yaml'

yaml_content = {
    'path': str(dataset_root.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 6,
    'names': {
        0: 'plastic',
        1: 'metal',
        2: 'cardboard',
        3: 'glass',
        4: 'paper',
        5: 'general_trash'
    }
}

with open(DATASET_YAML, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Configured runtime YAML at {DATASET_YAML}")
""")

    # Section 5
    add_markdown("""## Section 5: Baseline YOLO11s Validation Evaluation
Evaluate the baseline YOLO11s model on the exact same validation split (`conf=0.15`, `imgsz=640`).""")
    add_code("""from ultralytics import YOLO

# Check if baseline weights were attached as a dataset/model
baseline_weights = list(input_dir.glob('**/*weights_best*.pt'))
baseline_metrics = {}

if baseline_weights:
    baseline_path = str(baseline_weights[0])
    print(f"Found baseline weights: {baseline_path}")
    s_model = YOLO(baseline_path)
    s_val = s_model.val(data=DATASET_YAML, split='val', imgsz=640, conf=0.15, device=0, verbose=False)
    baseline_metrics = {
        'model': 'YOLO11s (Baseline Evaluated)',
        'imgsz': 640,
        'precision': float(s_val.results_dict.get('metrics/precision(B)', 0.0)),
        'recall': float(s_val.results_dict.get('metrics/recall(B)', 0.0)),
        'map50': float(s_val.results_dict.get('metrics/mAP50(B)', 0.0)),
        'map50_95': float(s_val.results_dict.get('metrics/mAP50-95(B)', 0.0)),
        'per_class_map50': [float(x) for x in s_val.box.ap50] if hasattr(s_val.box, 'ap50') else [],
    }
else:
    print("Baseline weights not attached. Using verified historical baseline metrics.")
    baseline_metrics = {
        'model': 'YOLO11s (Historical Baseline)',
        'imgsz': 640,
        'precision': 0.671,
        'recall': 0.519,
        'map50': 0.591,
        'map50_95': 0.446,
        'per_class_map50': [],
    }

print("Baseline Metrics:")
for k, v in baseline_metrics.items():
    if k != 'per_class_map50':
        print(f"  {k:15s}: {v}")
""")

    # Section 6
    add_markdown("""## Section 6: YOLO11m Training Execution with Safe Batch Fallback (16 -> 8 -> 4)
Hyperparameter Recipe:
- Architecture: YOLO11m Detection (`yolo11m.pt`)
- Input Size: $960 \\times 960$
- Max Epochs: 100
- Early Stopping: `patience=25` (monitored via Ultralytics fitness metric: `1.0 * mAP50-95`)
- Batch Size Strategy: Safe fallback (16 -> 8 -> 4). Gradient accumulation is handled natively by Ultralytics via nominal batch size `nbs=64`.
- Augmentation: `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.0`, `close_mosaic=15`
- Optimizer: `AdamW` (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`, `momentum=0.937`)
- Periodic Checkpoints: `save_period=5` (guarantees resume capability)
- Output Working Directory: `/kaggle/working/runs/train/yolo11m_waste_960/`""")
    add_code("""import gc

# Dynamic initial batch selection based on GPU memory
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
initial_batch = 16 if gpu_mem_gb >= 14.5 else 8
print(f"GPU VRAM: {gpu_mem_gb:.2f} GB -> Initial batch size: {initial_batch}")

model = YOLO('yolo11m.pt')

train_args = {
    'data': DATASET_YAML,
    'epochs': 100,
    'patience': 25,
    'batch': initial_batch,
    'imgsz': 960,
    'seed': 42,
    'device': 0,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'mosaic': 1.0,
    'close_mosaic': 15,
    'mixup': 0.15,
    'copy_paste': 0.0,
    'project': '/kaggle/working/runs/train',
    'name': 'yolo11m_waste_960',
    'save': True,
    'save_period': 5,
    'plots': True,
    'verbose': True
}

# Training Execution with Strict Batch Fallback (16 -> 8 -> 4)
actual_batch_used = initial_batch
try:
    print(f"Attempting training with batch={train_args['batch']}...")
    results = model.train(**train_args)
except Exception as e:
    if "out of memory" in str(e).lower() and train_args['batch'] > 8:
        print("[!] CUDA OOM detected at batch=16! Falling back to batch=8...")
        torch.cuda.empty_cache()
        gc.collect()
        train_args['batch'] = 8
        actual_batch_used = 8
        try:
            model = YOLO('yolo11m.pt')
            results = model.train(**train_args)
        except Exception as e2:
            if "out of memory" in str(e2).lower() and train_args['batch'] > 4:
                print("[!] CUDA OOM detected at batch=8! Falling back to batch=4...")
                torch.cuda.empty_cache()
                gc.collect()
                train_args['batch'] = 4
                actual_batch_used = 4
                model = YOLO('yolo11m.pt')
                results = model.train(**train_args)
            else:
                raise e2
    elif "out of memory" in str(e).lower() and train_args['batch'] > 4:
        print("[!] CUDA OOM detected! Falling back to batch=4...")
        torch.cuda.empty_cache()
        gc.collect()
        train_args['batch'] = 4
        actual_batch_used = 4
        model = YOLO('yolo11m.pt')
        results = model.train(**train_args)
    else:
        raise e

print(f"Training successfully completed with batch size: {actual_batch_used}")
""")

    # Section 7
    add_markdown("""## Section 7: Candidate Checkpoint Validation & Side-by-Side Comparison""")
    add_code("""best_weight = Path('/kaggle/working/runs/train/yolo11m_waste_960/weights/best.pt')
if not best_weight.exists():
    raise FileNotFoundError(f"Training completed but best.pt not found at {best_weight}!")

candidate_model = YOLO(str(best_weight))
m_val = candidate_model.val(data=DATASET_YAML, split='val', imgsz=960, conf=0.15, device=0, verbose=False)

candidate_metrics = {
    'model': 'YOLO11m (Candidate)',
    'imgsz': 960,
    'precision': float(m_val.results_dict.get('metrics/precision(B)', 0.0)),
    'recall': float(m_val.results_dict.get('metrics/recall(B)', 0.0)),
    'map50': float(m_val.results_dict.get('metrics/mAP50(B)', 0.0)),
    'map50_95': float(m_val.results_dict.get('metrics/mAP50-95(B)', 0.0)),
    'per_class_map50': [float(x) for x in m_val.box.ap50] if hasattr(m_val.box, 'ap50') else [],
    'per_class_map': [float(x) for x in m_val.box.ap] if hasattr(m_val.box, 'ap') else [],
}

print("=" * 75)
print(f"{'Metric':<25} | {'YOLO11s Baseline':<20} | {'YOLO11m Candidate':<20} | {'Delta':<10}")
print("=" * 75)

for metric_key, label in [
    ('precision', 'Precision (P)'),
    ('recall', 'Recall (R)'),
    ('map50', 'mAP@50'),
    ('map50_95', 'mAP@50-95'),
]:
    b_val = baseline_metrics.get(metric_key, 0.0)
    c_val = candidate_metrics.get(metric_key, 0.0)
    delta = c_val - b_val
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"{label:<25} | {b_val:<20.4f} | {c_val:<20.4f} | {delta_str:<10}")

print("=" * 75)
""")

    # Section 8
    add_markdown(r"""## Section 8: Dense Mixed-Waste Evaluation
Evaluate predictions on complex multi-object validation clusters ($\ge 4$ objects, $\ge 2$ classes).""")
    add_code("""import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash']
val_lbl_dir = dataset_root / 'labels' / 'val'
val_img_dir = dataset_root / 'images' / 'val'

dense_scenes = []
for lbl_file in val_lbl_dir.glob('*.txt'):
    classes_in_file = set()
    box_count = 0
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes_in_file.add(int(parts[0]))
                box_count += 1
    if box_count >= 4 and len(classes_in_file) >= 2:
        img_file = val_img_dir / (lbl_file.stem + '.jpg')
        if not img_file.exists():
            img_file = val_img_dir / (lbl_file.stem + '.png')
        if img_file.exists():
            dense_scenes.append((str(img_file), str(lbl_file), box_count, len(classes_in_file)))

print(f"Identified {len(dense_scenes)} dense mixed-waste scenes in validation set.")

# Plot sample predictions
sample_scenes = dense_scenes[:min(3, len(dense_scenes))]
fig, axes = plt.subplots(len(sample_scenes), 2, figsize=(16, 6 * len(sample_scenes)))
if len(sample_scenes) == 1:
    axes = np.expand_dims(axes, 0)

for idx, (img_p, lbl_p, b_cnt, c_cnt) in enumerate(sample_scenes):
    img = cv2.imread(img_p)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ground Truth
    axes[idx, 0].imshow(img_rgb)
    axes[idx, 0].set_title(f"Scene {idx+1}: Ground Truth ({b_cnt} items, {c_cnt} classes)")
    axes[idx, 0].axis('off')
    
    # YOLO11m Prediction
    res = candidate_model.predict(img_p, conf=0.15, imgsz=960, verbose=False)[0]
    res_bgr = res.plot()
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    
    axes[idx, 1].imshow(res_rgb)
    axes[idx, 1].set_title(f"YOLO11m Candidate ({len(res.boxes)} detections)")
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/dense_scene_comparison.png', dpi=150)
plt.close()
print("Saved dense_scene_comparison.png")
""")

    # Section 9
    add_markdown("""## Section 9: Checkpoint Export to Kaggle Working Directory
Place all candidate artifacts into `/kaggle/working/` for automated retrieval via `kaggle kernels output`.""")
    add_code("""import json

working_dir = Path('/kaggle/working')

# 1. Copy Best Model Checkpoint
shutil.copy(best_weight, working_dir / 'yolo11m_best.pt')
print(f"Exported candidate checkpoint: {working_dir / 'yolo11m_best.pt'} ({best_weight.stat().st_size / (1024**2):.2f} MB)")

# 2. Copy Training Plots & CSVs
train_run_dir = Path('/kaggle/working/runs/train/yolo11m_waste_960')
plot_files = ['results.csv', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'F1_curve.png', 'PR_curve.png']
for pf in plot_files:
    src_f = train_run_dir / pf
    if src_f.exists():
        shutil.copy(src_f, working_dir / pf)

# 3. Save Metrics Summary JSON
summary_payload = {
    'architecture': 'YOLO11m Detection',
    'imgsz': 960,
    'epochs_trained': 100,
    'actual_batch_used': actual_batch_used,
    'baseline_metrics': baseline_metrics,
    'candidate_metrics': candidate_metrics,
    'export_file': 'yolo11m_best.pt'
}

with open(working_dir / 'metrics_summary.json', 'w') as f:
    json.dump(summary_payload, f, indent=2)

print("Export package successfully assembled in /kaggle/working/!")
print("Antigravity can now pull outputs via: kaggle kernels output <kernel_id> -p models/experiments/")
""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("training/kaggle/ecovision_yolo11m_training.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1)

    print(f"Kaggle Notebook successfully written to {out_file.resolve()}")

if __name__ == "__main__":
    create_kaggle_notebook()
