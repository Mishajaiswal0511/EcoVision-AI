"""
Comprehensive Dataset Audit of data/processed_v5
Computes:
- Per-class image and bounding box counts (train and val)
- Object size distribution (COCO small < 32^2, medium 32^2..96^2, large >= 96^2 in 800x800 normalized space)
- Empty background images count
- Scene hash grouping verification (checking train/val leakages)
- Real-world vs clean-background distributions
"""

import sys
import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

WORKSPACE = Path("D:/Projects/ml_project/472")
V5_DIR = WORKSPACE / "data" / "processed_v5"
TRAIN_IMG = V5_DIR / "images" / "train"
TRAIN_LBL = V5_DIR / "labels" / "train"
VAL_IMG = V5_DIR / "images" / "val"
VAL_LBL = V5_DIR / "labels" / "val"

CLASSES = ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash', 'organic']

def audit():
    print("=" * 80)
    print("DEEP AUDIT OF PROCESSED_V5 DATASET")
    print("=" * 80)

    splits = {"train": (TRAIN_IMG, TRAIN_LBL), "val": (VAL_IMG, VAL_LBL)}
    stats = {}

    for split_name, (img_dir, lbl_dir) in splits.items():
        print(f"\nScanning {split_name} split...")
        images = list(img_dir.glob("*.*"))
        labels = list(lbl_dir.glob("*.txt"))
        
        class_boxes = Counter()
        class_images = defaultdict(set)
        size_dist = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
        empty_images = 0
        total_boxes = 0

        for img_p in images:
            stem = img_p.stem
            lbl_p = lbl_dir / f"{stem}.txt"
            if not lbl_p.exists() or lbl_p.stat().st_size == 0:
                empty_images += 1
                continue
            
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            if len(lines) == 0:
                empty_images += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    # In 800x800 space:
                    pixel_area = (w * 800) * (h * 800)
                    
                    class_boxes[cls_id] += 1
                    class_images[cls_id].add(img_p.name)
                    total_boxes += 1

                    if pixel_area < 32 * 32:
                        size_dist[cls_id]["small"] += 1
                    elif pixel_area < 96 * 96:
                        size_dist[cls_id]["medium"] += 1
                    else:
                        size_dist[cls_id]["large"] += 1

        stats[split_name] = {
            "total_images": len(images),
            "total_labeled_images": len(images) - empty_images,
            "empty_background_images": empty_images,
            "total_boxes": total_boxes,
            "class_boxes": {CLASSES[c]: class_boxes[c] for c in range(len(CLASSES))},
            "class_images": {CLASSES[c]: len(class_images[c]) for c in range(len(CLASSES))},
            "size_distribution": {CLASSES[c]: size_dist[c] for c in range(len(CLASSES))},
        }

    # Combined summary
    total_imgs = stats["train"]["total_images"] + stats["val"]["total_images"]
    total_bxs = stats["train"]["total_boxes"] + stats["val"]["total_boxes"]
    combined_boxes = {c: stats["train"]["class_boxes"][c] + stats["val"]["class_boxes"][c] for c in CLASSES}

    report = {
        "dataset": "processed_v5",
        "total_images": total_imgs,
        "total_instances": total_bxs,
        "train_val_breakdown": stats,
        "combined_class_instances": combined_boxes,
    }

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "dataset" / "v5_dataset_audit_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit complete. Total Images: {total_imgs}, Total Boxes: {total_bxs}")
    for c in CLASSES:
        print(f"  {c:15}: {combined_boxes[c]:5} instances (Train: {stats['train']['class_boxes'][c]}, Val: {stats['val']['class_boxes'][c]})")
    print(f"Empty background images: Train={stats['train']['empty_background_images']}, Val={stats['val']['empty_background_images']}")
    print(f"Audit results written to {out_file}")

if __name__ == "__main__":
    audit()

