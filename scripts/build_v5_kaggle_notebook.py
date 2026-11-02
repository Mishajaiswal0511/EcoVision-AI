"""
EcoVision AI - V5 Kaggle Training Notebook Builder
Generates training/kaggle_v5/ecovision_v5_training.ipynb for headless Kaggle GPU execution.
"""

import json
from pathlib import Path

WORKSPACE = Path("D:/Projects/ml_project/472")
NOTEBOOK_PATH = WORKSPACE / "training" / "kaggle_v5" / "ecovision_v5_training.ipynb"


def build_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + "\n" for l in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in text.strip().split("\n")]
        })

    # Header
    md(r"""# EcoVision AI V5 — Headless Kaggle GPU 7-Class Model Training
**Governing Standard:** MoEFCC / CPCB Solid Waste Management Rules 2026  
**Hardware Accelerator:** NVIDIA Tesla T4 / P100 (16 GB VRAM)  
**Taxonomy (7 Classes):** `0: plastic`, `1: metal`, `2: cardboard`, `3: glass`, `4: paper`, `5: general_trash`, `6: organic`  
**Dataset:** Scene-Grouped V5 Dataset (8,242 train / 3,007 val images, zero parent-scene leakage)  
**Experiments:**
- **Experiment A:** YOLO11s @ 640 (Standard controlled baseline on V5)
- **Experiment B:** YOLO11s @ 800 (Small-object optimized resolution for general_trash & litter)
""")

    # Cell 1: Environment & CUDA
    md("## Step 1: Verify Hardware Accelerator & GPU VRAM")
    code("""!nvidia-smi

import sys
import torch

print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CRITICAL ERROR: GPU is not available! Kaggle notebook accelerator must be set to GPU.")

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU Accelerator: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
""")

    # Cell 2: Dependencies
    md("## Step 2: Install Ultralytics and Supporting Libraries")
    code("""!pip install --quiet --upgrade "ultralytics>=8.3.0,<8.5.0" pyyaml pandas tabulate matplotlib seaborn

import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")
""")

    # Cell 3: Staging Dataset
    md("## Step 3: Unpack and Stage V5 Dataset in Fast Scratch Storage (`/tmp/dataset`)")
    code("""import os
import shutil
import zipfile
from pathlib import Path

scratch_dir = Path("/tmp/dataset/data/processed_v5")
scratch_dir.mkdir(parents=True, exist_ok=True)

input_dir = Path("/kaggle/input")
print(f"Scanning {input_dir}...")
for item in input_dir.iterdir():
    print(f"  Available input: {item.name}")

# Look for ecovision_v5_dataset.zip or uncompressed dataset
v5_zips = list(input_dir.glob("**/*v5*.zip")) + list(input_dir.glob("**/*dataset*.zip"))
if v5_zips:
    zip_path = v5_zips[0]
    print(f"Extracting {zip_path} to /tmp/dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/tmp/dataset")
    print("Extraction complete.")
else:
    # Check for uncompressed directory
    uncompressed = list(input_dir.glob("**/processed_v5"))
    if uncompressed:
        print(f"Copying uncompressed dataset from {uncompressed[0]}...")
        shutil.copytree(uncompressed[0], scratch_dir, dirs_exist_ok=True)
    else:
        raise FileNotFoundError("Could not locate V5 dataset in /kaggle/input/!")

# Verify dataset structure
train_imgs = list((scratch_dir / "images" / "train").glob("*.*"))
val_imgs = list((scratch_dir / "images" / "val").glob("*.*"))
print(f"Verified dataset in scratch: {len(train_imgs)} train images, {len(val_imgs)} val images.")

# Create runtime dataset.yaml with absolute /tmp paths
runtime_yaml_path = Path("/tmp/dataset/dataset_v5.yaml")
yaml_content = f\"\"\"path: {scratch_dir.as_posix()}
train: images/train
val: images/val

nc: 7
names: ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash', 'organic']
\"\"\"
runtime_yaml_path.write_text(yaml_content)
print(f"Wrote runtime YAML to {runtime_yaml_path}")
""")

    # Cell 4: Experiment A Training
    md("## Step 4: Experiment A — YOLO11s Standard Controlled Training @ 640")
    code("""from ultralytics import YOLO
import time

print("=" * 80)
print("STARTING EXPERIMENT A: YOLO11s @ 640 (V5 7-CLASS)")
print("=" * 80)

model_a = YOLO("yolo11s.pt")

t0 = time.time()
res_a = model_a.train(
    data="/tmp/dataset/dataset_v5.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.10,
    plots=True,
    project="/kaggle/working/runs",
    name="exp_a_yolo11s_640",
    exist_ok=True,
    verbose=True,
    seed=42
)
dur_a = time.time() - t0
print(f"Experiment A training completed in {dur_a / 60:.2f} minutes.")
""")

    # Cell 5: Validate Experiment A
    md("## Step 5: Validate Experiment A on V5 Validation Set")
    code("""val_res_a = model_a.val(
    data="/tmp/dataset/dataset_v5.yaml",
    split="val",
    imgsz=640,
    conf=0.15,
    iou=0.70,
    project="/kaggle/working/runs",
    name="val_exp_a",
    exist_ok=True
)

rd_a = val_res_a.results_dict
print("Experiment A Overall Metrics:")
print(f"  Precision: {rd_a.get('metrics/precision(B)', 0):.4f}")
print(f"  Recall   : {rd_a.get('metrics/recall(B)', 0):.4f}")
print(f"  mAP@50   : {rd_a.get('metrics/mAP50(B)', 0):.4f}")
print(f"  mAP@50-95: {rd_a.get('metrics/mAP50-95(B)', 0):.4f}")
""")

    # Cell 6: Experiment B Training
    md("## Step 6: Experiment B — YOLO11s Small-Object Optimized @ 800")
    code("""print("=" * 80)
print("STARTING EXPERIMENT B: YOLO11s @ 800 (SMALL-OBJECT OPTIMIZATION)")
print("=" * 80)

model_b = YOLO("yolo11s.pt")

t0 = time.time()
res_b = model_b.train(
    data="/tmp/dataset/dataset_v5.yaml",
    epochs=50,
    imgsz=800,
    batch=12,
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.15,
    plots=True,
    project="/kaggle/working/runs",
    name="exp_b_yolo11s_800",
    exist_ok=True,
    verbose=True,
    seed=42
)
dur_b = time.time() - t0
print(f"Experiment B training completed in {dur_b / 60:.2f} minutes.")
""")

    # Cell 7: Validate Experiment B
    md("## Step 7: Validate Experiment B on V5 Validation Set")
    code("""val_res_b = model_b.val(
    data="/tmp/dataset/dataset_v5.yaml",
    split="val",
    imgsz=800,
    conf=0.15,
    iou=0.70,
    project="/kaggle/working/runs",
    name="val_exp_b",
    exist_ok=True
)

rd_b = val_res_b.results_dict
print("Experiment B Overall Metrics:")
print(f"  Precision: {rd_b.get('metrics/precision(B)', 0):.4f}")
print(f"  Recall   : {rd_b.get('metrics/recall(B)', 0):.4f}")
print(f"  mAP@50   : {rd_b.get('metrics/mAP50(B)', 0):.4f}")
print(f"  mAP@50-95: {rd_b.get('metrics/mAP50-95(B)', 0):.4f}")
""")

    # Cell 8: Comparison & Export
    md("## Step 8: Side-by-Side Comparison & Artifact Export")
    code("""import json
import shutil
from tabulate import tabulate

classes = ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash', 'organic']

# Extract per-class
box_a = val_res_a.box
box_b = val_res_b.box

table_data = []
for i, c in enumerate(classes):
    p_a, r_a, ap50_a = box_a.p[i], box_a.r[i], box_a.ap50[i]
    p_b, r_b, ap50_b = box_b.p[i], box_b.r[i], box_b.ap50[i]
    table_data.append([c, f"{ap50_a:.4f}", f"{r_a:.4f}", f"{ap50_b:.4f}", f"{r_b:.4f}", f"{ap50_b - ap50_a:+.4f}"])

headers = ["Class", "Exp A mAP50 (640)", "Exp A Recall", "Exp B mAP50 (800)", "Exp B Recall", "Delta mAP50"]
print(tabulate(table_data, headers=headers, tablefmt="github"))

# Determine best candidate
map_a = rd_a.get('metrics/mAP50(B)', 0)
map_b = rd_b.get('metrics/mAP50(B)', 0)

best_name = "Exp B (800)" if map_b >= map_a else "Exp A (640)"
best_pt = Path("/kaggle/working/runs/exp_b_yolo11s_800/weights/best.pt") if map_b >= map_a else Path("/kaggle/working/runs/exp_a_yolo11s_640/weights/best.pt")

output_best = Path("/kaggle/working/yolo11s_v5_best.pt")
if best_pt.exists():
    shutil.copy2(best_pt, output_best)
    print(f"Copied best checkpoint to {output_best} ({output_best.stat().st_size / (1024**2):.2f} MB)")

# Save metrics JSON
export_metrics = {
    "exp_a_640": rd_a,
    "exp_b_800": rd_b,
    "best_candidate": best_name
}
with open("/kaggle/working/v5_evaluation_metrics.json", "w") as f:
    json.dump(export_metrics, f, indent=2)

print("TRAINING AND BENCHMARK COMPLETE.")
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated Kaggle training notebook at {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()

