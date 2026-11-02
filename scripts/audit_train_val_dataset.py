"""
EcoVision AI - Comprehensive Dataset Audit Script
Performs a deep diagnostic analysis of approved train and validation splits:
- Image counts, instance counts per class
- Class imbalance & frequency distributions
- Object size & aspect ratio distribution (small, medium, large, extreme aspect ratio)
- Crowded scenes distribution
- Dataset source breakdown (TACO, General Waste, Clean Backgrounds)
- Duplicate / near-duplicate leakage detection (file hash & base-stem matching)
- Annotation anomalies (out of bounds, zero area, degenerate coordinates)
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASS_NAMES = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]

DATA_ROOT = Path("data/processed_v3")
TRAIN_IMG = DATA_ROOT / "images" / "train"
TRAIN_LBL = DATA_ROOT / "labels" / "train"
VAL_IMG = DATA_ROOT / "images" / "val"
VAL_LBL = DATA_ROOT / "labels" / "val"

def hash_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def analyze_split(img_dir: Path, lbl_dir: Path, split_name: str):
    print(f"Analyzing {split_name} split...")
    images = list(img_dir.glob("*.*"))
    labels = list(lbl_dir.glob("*.txt"))

    img_stems = {p.stem: p for p in images}
    lbl_stems = {p.stem: p for p in labels}

    missing_labels = [s for s in img_stems if s not in lbl_stems]
    missing_images = [s for s in lbl_stems if s not in img_stems]

    class_counts = Counter()
    obj_counts_per_img = []
    box_areas = []
    aspect_ratios = []
    small_objects = 0      # area < 0.005
    medium_objects = 0     # 0.005 <= area < 0.05
    large_objects = 0      # area >= 0.05
    extreme_ar_boxes = 0   # ar > 4 or ar < 0.25

    anomalies = []
    source_counts = Counter()
    empty_background_images = 0

    hashes = {}

    for stem, img_path in img_stems.items():
        # Check source
        if stem.startswith("taco_"):
            source_counts["taco"] += 1
        elif stem.startswith("gw_"):
            source_counts["general_waste"] += 1
        else:
            source_counts["other_background"] += 1

        lbl_path = lbl_stems.get(stem)
        if not lbl_path or lbl_path.stat().st_size == 0:
            empty_background_images += 1
            obj_counts_per_img.append(0)
            continue

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_background_images += 1
            obj_counts_per_img.append(0)
            continue

        obj_counts_per_img.append(len(lines))

        for line_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                anomalies.append(f"{lbl_path.name}:{line_idx}: invalid token count ({len(parts)})")
                continue
            try:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                anomalies.append(f"{lbl_path.name}:{line_idx}: non-float coordinates")
                continue

            if not (0 <= cid < len(CLASS_NAMES)):
                anomalies.append(f"{lbl_path.name}:{line_idx}: invalid class id {cid}")
                continue

            # Check bbox bounds
            if xc < 0 or xc > 1 or yc < 0 or yc > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                anomalies.append(f"{lbl_path.name}:{line_idx}: box bounds anomaly (xc={xc}, yc={yc}, w={w}, h={h})")

            area = w * h
            ar = w / h if h > 0 else 0

            class_counts[CLASS_NAMES[cid]] += 1
            box_areas.append(area)
            aspect_ratios.append(ar)

            if area < 0.005:
                small_objects += 1
            elif area < 0.05:
                medium_objects += 1
            else:
                large_objects += 1

            if ar > 4.0 or ar < 0.25:
                extreme_ar_boxes += 1

    total_instances = sum(class_counts.values())

    return {
        "split": split_name,
        "total_images": len(images),
        "total_labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_background_images": empty_background_images,
        "total_instances": total_instances,
        "class_counts": dict(class_counts),
        "class_percentages": {c: round(class_counts[c] / total_instances * 100, 2) if total_instances > 0 else 0 for c in CLASS_NAMES},
        "source_breakdown": dict(source_counts),
        "objects_per_image": {
            "mean": round(float(np.mean(obj_counts_per_img)), 2) if obj_counts_per_img else 0,
            "median": float(np.median(obj_counts_per_img)) if obj_counts_per_img else 0,
            "p95": float(np.percentile(obj_counts_per_img, 95)) if obj_counts_per_img else 0,
            "max": int(np.max(obj_counts_per_img)) if obj_counts_per_img else 0,
            "single_object_images": sum(1 for n in obj_counts_per_img if n == 1),
            "multi_2_3_images": sum(1 for n in obj_counts_per_img if 2 <= n <= 3),
            "dense_4_plus_images": sum(1 for n in obj_counts_per_img if n >= 4),
        },
        "object_size_distribution": {
            "small_under_0_005": small_objects,
            "small_pct": round(small_objects / total_instances * 100, 2) if total_instances > 0 else 0,
            "medium_0_005_to_0_05": medium_objects,
            "medium_pct": round(medium_objects / total_instances * 100, 2) if total_instances > 0 else 0,
            "large_above_0_05": large_objects,
            "large_pct": round(large_objects / total_instances * 100, 2) if total_instances > 0 else 0,
            "extreme_aspect_ratio_boxes": extreme_ar_boxes,
            "extreme_ar_pct": round(extreme_ar_boxes / total_instances * 100, 2) if total_instances > 0 else 0,
        },
        "anomalies_count": len(anomalies),
        "sample_anomalies": anomalies[:10],
    }

def check_leakage():
    print("Checking for duplicate image leakage between train and validation...")
    train_images = list(TRAIN_IMG.glob("*.*"))
    val_images = list(VAL_IMG.glob("*.*"))

    train_hashes = {}
    for p in train_images:
        train_hashes[hash_file(p)] = p.name

    exact_duplicates = []
    for p in val_images:
        vh = hash_file(p)
        if vh in train_hashes:
            exact_duplicates.append({"val_image": p.name, "train_image": train_hashes[vh]})

    # Check stem-level leakage (e.g. Roboflow augmented clones with _jpg.rf.xxxx)
    def extract_base_stem(filename):
        # e.g., gw_plastic123_jpg.rf.abc1234 -> gw_plastic123
        parts = filename.split(".rf.")
        if len(parts) > 1:
            base = parts[0]
            for ext in ["_jpg", "_jpeg", "_png", "_JPG", "_JPEG", "_PNG"]:
                if base.endswith(ext):
                    base = base[:-len(ext)]
            return base
        return filename.split(".")[0]

    train_bases = defaultdict(list)
    for p in train_images:
        train_bases[extract_base_stem(p.name)].append(p.name)

    base_stem_overlap = []
    for p in val_images:
        v_base = extract_base_stem(p.name)
        if v_base in train_bases:
            base_stem_overlap.append({"val_image": p.name, "base_stem": v_base, "train_matches": train_bases[v_base][:3]})

    return {
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "exact_content_duplicates": len(exact_duplicates),
        "sample_exact_duplicates": exact_duplicates[:5],
        "base_stem_overlap_count": len(base_stem_overlap),
        "sample_base_stem_overlap": base_stem_overlap[:5],
    }

def main():
    print("=" * 80)
    print("ECOVISION AI - COMPREHENSIVE DATASET & ANNOTATION AUDIT")
    print("=" * 80)

    train_stats = analyze_split(TRAIN_IMG, TRAIN_LBL, "train")
    val_stats = analyze_split(VAL_IMG, VAL_LBL, "val")
    leakage = check_leakage()

    output = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "leakage_analysis": leakage,
    }

    out_file = Path("models/experiments/dataset_comprehensive_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nAudit complete. JSON saved to {out_file}")

if __name__ == "__main__":
    main()

