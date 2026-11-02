"""
EcoVision AI - General Trash Diagnostic Script
Read-only analysis of general_trash (class 5) in data/processed_v4/
Strictly read-only: does not modify any source images, labels, models, or code.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASSES = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]
COLOR_PALETTE = {
    0: (0, 165, 255),    # plastic: orange
    1: (255, 255, 0),    # metal: cyan
    2: (42, 42, 165),    # cardboard: brown
    3: (0, 255, 0),      # glass: green
    4: (255, 0, 255),    # paper: magenta
    5: (0, 0, 255),      # general_trash: red/crimson
}

DATASET_DIR = Path("data/processed_v4")
OUTPUT_DIR = Path("models/experiments/general_trash_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_percentiles(arr):
    if len(arr) == 0:
        return {}
    p = np.percentile(arr, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {
        "min": float(p[0]),
        "p01": float(p[1]),
        "p05": float(p[2]),
        "p10": float(p[3]),
        "p25": float(p[4]),
        "median": float(p[5]),
        "p75": float(p[6]),
        "p90": float(p[7]),
        "p95": float(p[8]),
        "p99": float(p[9]),
        "max": float(p[10]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def analyze_split(split: str):
    label_dir = DATASET_DIR / "labels" / split
    img_dir = DATASET_DIR / "images" / split
    
    total_imgs = len(list(img_dir.glob("*.*")))
    
    class_instances = Counter()
    class_images = defaultdict(set)
    
    gt_instances_per_img = []
    
    gt_boxes = {"w": [], "h": [], "area": [], "ar": []}
    all_class_areas = {c: [] for c in CLASSES}
    
    co_occur = Counter()
    
    images_with_gt = []
    
    for f in label_dir.glob("*.txt"):
        stem = f.stem
        lines = [l.strip().split() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        img_classes = set()
        c5_count = 0
        
        for p in lines:
            cid = int(p[0])
            w, h = float(p[3]), float(p[4])
            class_instances[cid] += 1
            img_classes.add(cid)
            all_class_areas[CLASSES[cid]].append(w * h)
            
            if cid == 5:
                c5_count += 1
                gt_boxes["w"].append(w)
                gt_boxes["h"].append(h)
                gt_boxes["area"].append(w * h)
                gt_boxes["ar"].append(w / max(h, 1e-6))
                
        for cid in img_classes:
            class_images[cid].add(stem)
            
        if 5 in img_classes:
            gt_instances_per_img.append(c5_count)
            images_with_gt.append((stem, c5_count, img_classes))
            for cid in img_classes:
                if cid != 5:
                    co_occur[CLASSES[cid]] += 1

    return {
        "split": split,
        "total_images": total_imgs,
        "class_instances": {CLASSES[i]: class_instances[i] for i in range(6)},
        "class_images_count": {CLASSES[i]: len(class_images[i]) for i in range(6)},
        "general_trash_images": len(class_images[5]),
        "general_trash_instances": class_instances[5],
        "general_trash_per_image_stats": compute_percentiles(gt_instances_per_img),
        "general_trash_per_image_counts": gt_instances_per_img,
        "box_stats": {
            "w": compute_percentiles(gt_boxes["w"]),
            "h": compute_percentiles(gt_boxes["h"]),
            "area": compute_percentiles(gt_boxes["area"]),
            "ar": compute_percentiles(gt_boxes["ar"]),
            "raw_areas": gt_boxes["area"],
            "raw_ars": gt_boxes["ar"],
            "raw_w": gt_boxes["w"],
            "raw_h": gt_boxes["h"],
        },
        "all_class_areas": all_class_areas,
        "co_occurrence_with_gt": dict(co_occur),
        "gt_image_list": images_with_gt,
    }


def draw_bounding_boxes(img_path: Path, label_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h_img, w_img = img.shape[:2]
    
    if label_path.exists():
        lines = [l.strip().split() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Draw other classes first, then general_trash on top
        lines_sorted = sorted(lines, key=lambda p: 1 if int(p[0]) == 5 else 0)
        
        for p in lines_sorted:
            cid = int(p[0])
            cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)
            
            color = COLOR_PALETTE.get(cid, (255, 255, 255))
            thickness = 3 if cid == 5 else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            label_text = f"{CLASSES[cid]}"
            if cid == 5:
                label_text += f" ({bw*w_img:.0f}x{bh*h_img:.0f}px)"
                
            font_scale = 0.4 if cid != 5 else 0.5
            cv2.putText(img, label_text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            
    return img


def create_contact_sheet(train_data, val_data, output_path: Path):
    print("Generating contact sheet...")
    # Select 12 representative images:
    # 6 from train: 1 single cigarette, 1 multi cigarette, 1 dense litter, 1 co-occur with plastic, 1 small butt, 1 rope/unlabeled
    # 6 from val: 1 single cigarette, 1 multi cigarette, 1 dense litter, 1 co-occur with plastic, 1 small butt, 1 rope/unlabeled
    
    train_imgs = train_data["gt_image_list"]
    val_imgs = val_data["gt_image_list"]
    
    train_by_count = sorted(train_imgs, key=lambda x: x[1])
    val_by_count = sorted(val_imgs, key=lambda x: x[1])
    
    # Pick diverse selections:
    selected_train = [
        train_by_count[0][0],   # single instance
        train_by_count[50][0],  # single instance
        train_by_count[250][0], # 2 instances
        train_by_count[400][0], # 4-5 instances
        train_by_count[-5][0],  # high density (~20-35)
        train_by_count[-1][0],  # extreme density (51)
    ]
    
    selected_val = [
        val_by_count[0][0],     # single instance
        val_by_count[25][0],    # single instance
        val_by_count[80][0],    # 2 instances
        val_by_count[120][0],   # 3-4 instances
        val_by_count[-4][0],    # high density (~16-17)
        val_by_count[-1][0],    # extreme density (28)
    ]
    
    selected = [("train", name) for name in selected_train] + [("val", name) for name in selected_val]
    
    tiles = []
    target_w, target_h = 360, 270
    
    for split, stem in selected:
        img_path = list((DATASET_DIR / "images" / split).glob(f"{stem}.*"))[0]
        lbl_path = DATASET_DIR / "labels" / split / f"{stem}.txt"
        
        rendered = draw_bounding_boxes(img_path, lbl_path)
        if rendered is not None:
            # Add header banner
            h_r, w_r = rendered.shape[:2]
            header = np.zeros((30, w_r, 3), dtype=np.uint8)
            header[:] = (30, 30, 30)
            
            # Count c5 in label
            lines = [l.strip().split() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            c5_cnt = sum(1 for p in lines if int(p[0]) == 5)
            other_cnt = len(lines) - c5_cnt
            
            text = f"[{split.upper()}] {stem[:22]}... | GT: {c5_cnt} trash, {other_cnt} others"
            cv2.putText(header, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            combined = np.vstack([header, rendered])
            resized = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)
            tiles.append(resized)
            
    # Combine into 4 rows x 3 columns grid
    rows = []
    for r in range(4):
        row_tiles = tiles[r*3 : (r+1)*3]
        rows.append(np.hstack(row_tiles))
    grid = np.vstack(rows)
    
    # Save contact sheet
    cv2.imwrite(str(output_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Contact sheet saved to {output_path} ({grid.shape[1]}x{grid.shape[0]} px)")


def main():
    print("=" * 80)
    print("ECOVISION AI - GENERAL TRASH (CLASS 5) DIAGNOSTIC AUDIT")
    print("=" * 80)
    
    train_res = analyze_split("train")
    val_res = analyze_split("val")
    
    # Build class_distribution.json
    total_imgs = train_res["total_images"] + val_res["total_images"]
    class_dist = {
        "dataset": "data/processed_v4/",
        "total_images": total_imgs,
        "splits": {
            "train": {
                "total_images": train_res["total_images"],
                "images_with_general_trash": train_res["general_trash_images"],
                "pct_images_with_general_trash": round(train_res["general_trash_images"] / train_res["total_images"] * 100, 2),
                "total_instances": sum(train_res["class_instances"].values()),
                "class_instances": train_res["class_instances"],
                "class_instance_shares": {c: round(cnt / sum(train_res["class_instances"].values()) * 100, 2) for c, cnt in train_res["class_instances"].items()},
                "class_images_count": train_res["class_images_count"],
                "class_to_trash_ratio": {c: round(cnt / max(train_res["class_instances"]["general_trash"], 1), 2) for c, cnt in train_res["class_instances"].items()},
            },
            "val": {
                "total_images": val_res["total_images"],
                "images_with_general_trash": val_res["general_trash_images"],
                "pct_images_with_general_trash": round(val_res["general_trash_images"] / val_res["total_images"] * 100, 2),
                "total_instances": sum(val_res["class_instances"].values()),
                "class_instances": val_res["class_instances"],
                "class_instance_shares": {c: round(cnt / sum(val_res["class_instances"].values()) * 100, 2) for c, cnt in val_res["class_instances"].items()},
                "class_images_count": val_res["class_images_count"],
                "class_to_trash_ratio": {c: round(cnt / max(val_res["class_instances"]["general_trash"], 1), 2) for c, cnt in val_res["class_instances"].items()},
            },
            "combined": {
                "total_images": total_imgs,
                "general_trash_images": train_res["general_trash_images"] + val_res["general_trash_images"],
                "total_general_trash_instances": train_res["general_trash_instances"] + val_res["general_trash_instances"],
                "total_all_instances": sum(train_res["class_instances"].values()) + sum(val_res["class_instances"].values()),
                "general_trash_overall_share": round((train_res["general_trash_instances"] + val_res["general_trash_instances"]) / (sum(train_res["class_instances"].values()) + sum(val_res["class_instances"].values())) * 100, 2),
            }
        },
        "co_occurrence_in_images_with_general_trash": {
            "train": {
                "general_trash_total_images": train_res["general_trash_images"],
                "co_occurring_classes": train_res["co_occurrence_with_gt"],
                "co_occurring_pct": {c: round(cnt / train_res["general_trash_images"] * 100, 2) for c, cnt in train_res["co_occurrence_with_gt"].items()},
                "pure_general_trash_images": train_res["general_trash_images"] - max(train_res["co_occurrence_with_gt"].values()) if train_res["co_occurrence_with_gt"] else 0
            },
            "val": {
                "general_trash_total_images": val_res["general_trash_images"],
                "co_occurring_classes": val_res["co_occurrence_with_gt"],
                "co_occurring_pct": {c: round(cnt / val_res["general_trash_images"] * 100, 2) for c, cnt in val_res["co_occurrence_with_gt"].items()},
                "pure_general_trash_images": val_res["general_trash_images"] - max(val_res["co_occurrence_with_gt"].values()) if val_res["co_occurrence_with_gt"] else 0
            }
        }
    }
    
    with open(OUTPUT_DIR / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump(class_dist, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'class_distribution.json'}")

    # Build box_statistics.json
    def box_size_breakdown(areas):
        arr = np.array(areas)
        n = len(arr)
        return {
            "under_0_0001 (micro < 0.01% img)": int(np.sum(arr < 0.0001)),
            "pct_under_0_0001": round(float(np.mean(arr < 0.0001) * 100), 2),
            "under_0_0005 (< 0.05% img)": int(np.sum(arr < 0.0005)),
            "pct_under_0_0005": round(float(np.mean(arr < 0.0005) * 100), 2),
            "under_0_0010 (< 0.1% img)": int(np.sum(arr < 0.001)),
            "pct_under_0_0010": round(float(np.mean(arr < 0.001) * 100), 2),
            "under_0_0050 (< 0.5% img)": int(np.sum(arr < 0.005)),
            "pct_under_0_0050": round(float(np.mean(arr < 0.005) * 100), 2),
            "under_0_0100 (< 1% img)": int(np.sum(arr < 0.01)),
            "pct_under_0_0100": round(float(np.mean(arr < 0.01) * 100), 2),
            "large_over_0_0500 (> 5% img)": int(np.sum(arr > 0.05)),
            "pct_over_0_0500": round(float(np.mean(arr > 0.05) * 100), 2),
        }

    def aspect_ratio_breakdown(ars):
        arr = np.array(ars)
        return {
            "extreme_tall_ar_under_0_25": int(np.sum(arr < 0.25)),
            "pct_extreme_tall": round(float(np.mean(arr < 0.25) * 100), 2),
            "tall_ar_0_25_to_0_50": int(np.sum((arr >= 0.25) & (arr < 0.5))),
            "pct_tall": round(float(np.mean((arr >= 0.25) & (arr < 0.5)) * 100), 2),
            "balanced_ar_0_50_to_2_00": int(np.sum((arr >= 0.5) & (arr <= 2.0))),
            "pct_balanced": round(float(np.mean((arr >= 0.5) & (arr <= 2.0)) * 100), 2),
            "wide_ar_2_00_to_4_00": int(np.sum((arr > 2.0) & (arr <= 4.0))),
            "pct_wide": round(float(np.mean((arr > 2.0) & (arr <= 4.0)) * 100), 2),
            "extreme_wide_ar_over_4_00": int(np.sum(arr > 4.0)),
            "pct_extreme_wide": round(float(np.mean(arr > 4.0) * 100), 2),
        }

    def density_breakdown(counts):
        arr = np.array(counts)
        return {
            "single_instance (1)": int(np.sum(arr == 1)),
            "pct_single": round(float(np.mean(arr == 1) * 100), 2),
            "sparse (2-3)": int(np.sum((arr >= 2) & (arr <= 3))),
            "pct_sparse": round(float(np.mean((arr >= 2) & (arr <= 3)) * 100), 2),
            "moderate (4-9)": int(np.sum((arr >= 4) & (arr <= 9))),
            "pct_moderate": round(float(np.mean((arr >= 4) & (arr <= 9)) * 100), 2),
            "crowded_10_plus": int(np.sum(arr >= 10)),
            "pct_crowded": round(float(np.mean(arr >= 10) * 100), 2),
            "max_instances_in_single_image": int(np.max(arr)) if len(arr) else 0,
        }

    # Class-wise size comparison in val
    all_class_comparison = {}
    for c in CLASSES:
        val_c_areas = np.array(val_res["all_class_areas"][c])
        all_class_comparison[c] = {
            "count": len(val_c_areas),
            "median_area": round(float(np.median(val_c_areas)), 6),
            "mean_area": round(float(np.mean(val_c_areas)), 6),
            "pct_area_under_0_001": round(float(np.mean(val_c_areas < 0.001) * 100), 2),
            "pct_area_under_0_005": round(float(np.mean(val_c_areas < 0.005) * 100), 2),
            "median_pixel_dim_at_640": f"{np.sqrt(np.median(val_c_areas))*640:.1f} x {np.sqrt(np.median(val_c_areas))*640:.1f} px",
        }

    box_stats = {
        "dataset": "data/processed_v4/",
        "general_trash_box_statistics": {
            "train": {
                "instances_count": train_res["general_trash_instances"],
                "width_normalized": train_res["box_stats"]["w"],
                "height_normalized": train_res["box_stats"]["h"],
                "area_normalized": train_res["box_stats"]["area"],
                "aspect_ratio": train_res["box_stats"]["ar"],
                "size_bins": box_size_breakdown(train_res["box_stats"]["raw_areas"]),
                "aspect_ratio_bins": aspect_ratio_breakdown(train_res["box_stats"]["raw_ars"]),
                "instances_per_image_bins": density_breakdown(train_res["general_trash_per_image_counts"]),
            },
            "val": {
                "instances_count": val_res["general_trash_instances"],
                "width_normalized": val_res["box_stats"]["w"],
                "height_normalized": val_res["box_stats"]["h"],
                "area_normalized": val_res["box_stats"]["area"],
                "aspect_ratio": val_res["box_stats"]["ar"],
                "size_bins": box_size_breakdown(val_res["box_stats"]["raw_areas"]),
                "aspect_ratio_bins": aspect_ratio_breakdown(val_res["box_stats"]["raw_ars"]),
                "instances_per_image_bins": density_breakdown(val_res["general_trash_per_image_counts"]),
            }
        },
        "macro_vs_micro_class_comparison_val": all_class_comparison
    }

    with open(OUTPUT_DIR / "box_statistics.json", "w", encoding="utf-8") as f:
        json.dump(box_stats, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'box_statistics.json'}")

    # Generate Contact Sheet
    contact_sheet_path = OUTPUT_DIR / "contact_sheet.jpg"
    create_contact_sheet(train_res, val_res, contact_sheet_path)

    print("\nALL STATISTICAL ARTIFACTS GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

