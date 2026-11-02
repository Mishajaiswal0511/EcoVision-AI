"""
EcoVision AI - V5 Dataset Packager
Builds data/processed_v5/ with 7-class disposal-oriented taxonomy:
  0: plastic
  1: metal
  2: cardboard
  3: glass
  4: paper
  5: general_trash
  6: organic

Strictly read-only with respect to V3, V4, and models/weights_best.pt.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path("D:/Projects/ml_project/472")
V4_DIR = WORKSPACE / "data" / "processed_v4"
V5_DIR = WORKSPACE / "data" / "processed_v5"
RAW_GWD = WORKSPACE / "data" / "raw" / "general_waste_data"
RAW_TACO = WORKSPACE / "data" / "raw" / "garbage_taco-official-all-class_v6"

EXP_DIR = WORKSPACE / "models" / "experiments" / "v5_taxonomy"
EXP_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_V5 = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash", "organic"]

# TACO 59-class to V5 mapping
TACO_TO_V5 = {
    # 0: plastic
    "Clear plastic bottle": 0,
    "Other plastic bottle": 0,
    "Disposable plastic cup": 0,
    "Plastic lid": 0,
    "Plastic bottle cap": 0,
    "Spread tub": 0,
    "Tupperware": 0,
    "Other plastic container": 0,
    "Other plastic cup": 0,
    "Other plastic": 0,
    "Plastic film": 0,
    "Single-use carrier bag": 0,
    "Polypropylene bag": 0,
    "Six pack rings": 0,
    "Garbage bag": 0,
    "Other plastic wrapper": 0,
    # 1: metal
    "Drink can": 1,
    "Food Can": 1,
    "Metal bottle cap": 1,
    "Metal lid": 1,
    "Pop tab": 1,
    "Aerosol": 1,
    "Scrap metal": 1,
    "Aluminium foil": 1,
    # 2: cardboard
    "Corrugated carton": 2,
    "Egg carton": 2,
    "Toilet tube": 2,
    "Other carton": 2,
    "Drink carton": 2,
    # 3: glass
    "Glass bottle": 3,
    "Glass jar": 3,
    "Glass cup": 3,
    "Broken glass": 3,
    # 4: paper
    "Normal paper": 4,
    "Magazine paper": 4,
    "Paper bag": 4,
    "Paper straw": 4,
    "Wrapping paper": 4,
    "Paper cup": 4,
    # 5: general_trash (residual, non-recyclables, composites, contamination)
    "Cigarette": 5,
    "Rope - strings": 5,
    "Aluminium blister pack": 5,
    "Carded blister pack": 5,
    "Unlabeled litter": 5,
    "Crisp packet": 5,
    "Foam cup": 5,
    "Foam food container": 5,
    "Styrofoam piece": 5,
    "Plastic straw": 5,
    "Plastic utensils": 5,
    "Squeezable tube": 5,
    "Plastic glooves": 5,
    "Pizza box": 5,
    "Meal carton": 5,
    "Disposable food container": 5,
    "Tissues": 5,
    # 6: organic (wet waste / food waste)
    "Food waste": 6,
    # Excluded
    "Battery": None,
    "Shoe": None,
}

# GWD mapping to V5:
# 0: BIODEGRADABLE -> 6 (organic)
# 1: CARDBOARD -> 2 (cardboard)
# 2: GLASS -> 3 (glass)
# 3: METAL -> 1 (metal)
# 4: PAPER -> 4 (paper)
# 5: PLASTIC -> 0 (plastic)
GWD_TO_V5 = {
    0: 6, # organic
    1: 2, # cardboard
    2: 3, # glass
    3: 1, # metal
    4: 4, # paper
    5: 0, # plastic
}


def load_taco_raw_dict():
    print("Indexing raw TACO label files...")
    taco_dict = {}
    for p in RAW_TACO.rglob("*.txt"):
        if "labels" in p.parts:
            taco_dict[p.name] = p
    print(f"Indexed {len(taco_dict)} raw TACO label files.")
    return taco_dict


def load_gwd_raw_dict():
    print("Indexing raw GWD label files...")
    gwd_dict = {}
    for p in RAW_GWD.rglob("*.txt"):
        if "labels" in p.parts:
            gwd_dict[p.name] = p
    print(f"Indexed {len(gwd_dict)} raw GWD label files.")
    return gwd_dict


def load_taco_class_names():
    with open(RAW_TACO / "data.yaml", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    return cfg["names"]


def parse_polygon_or_box(parts):
    # If standard 5 tokens: class cx cy w h
    if len(parts) == 5:
        return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    # If polygon: class x1 y1 x2 y2 ... xn yn
    coords = [float(x) for x in parts[1:]]
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
        ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        return cx, cy, w, h
    return None


def main():
    print("=" * 80)
    print("PACKAGING ECOVISION AI V5 (7-CLASS DISPOSAL-ORIENTED DATASET)")
    print("=" * 80)

    # 1. Prepare target directories
    for split in ["train", "val"]:
        (V5_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (V5_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    taco_files = load_taco_raw_dict()
    gwd_files = load_gwd_raw_dict()
    taco_names = load_taco_class_names()

    split_manifest = {"train": [], "val": []}
    class_stats = {"train": Counter(), "val": Counter()}

    for split in ["train", "val"]:
        print(f"\nProcessing {split} split from V4 source images...")
        v4_img_dir = V4_DIR / "images" / split
        img_files = sorted(list(v4_img_dir.glob("*.*")))
        print(f"Found {len(img_files)} images in {split}.")

        for img_p in img_files:
            stem = img_p.stem
            name = img_p.name
            target_img_p = V5_DIR / "images" / split / name
            target_lbl_p = V5_DIR / "labels" / split / f"{stem}.txt"

            # Hardlink image
            if not target_img_p.exists():
                try:
                    os.link(img_p, target_img_p)
                except OSError:
                    shutil.copy2(img_p, target_img_p)

            # Generate label
            valid_boxes = []

            if name.startswith("neg_"):
                # Negative background image: 0-byte label
                pass

            elif name.startswith("gw_"):
                # GWD image
                raw_name = name.replace("gw_", "", 1)
                raw_lbl_name = f"{Path(raw_name).stem}.txt"
                if raw_lbl_name in gwd_files:
                    lines = gwd_files[raw_lbl_name].read_text(encoding="utf-8").splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        raw_cid = int(parts[0])
                        v5_cid = GWD_TO_V5.get(raw_cid)
                        if v5_cid is not None:
                            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 0.003 and h > 0.003:
                                valid_boxes.append((v5_cid, cx, cy, w, h))

            elif name.startswith("taco_"):
                # TACO image
                raw_name = name.replace("taco_", "", 1)
                raw_lbl_name = f"{Path(raw_name).stem}.txt"
                if raw_lbl_name in taco_files:
                    lines = taco_files[raw_lbl_name].read_text(encoding="utf-8").splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        raw_cid = int(parts[0])
                        if raw_cid < len(taco_names):
                            cname = taco_names[raw_cid]
                            v5_cid = TACO_TO_V5.get(cname)
                            if v5_cid is not None:
                                box = parse_polygon_or_box(parts)
                                if box is not None:
                                    cx, cy, w, h = box
                                    if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 0.003 and h > 0.003:
                                        valid_boxes.append((v5_cid, cx, cy, w, h))

            # Write label file
            with open(target_lbl_p, "w", encoding="utf-8") as lf:
                for v5_cid, cx, cy, w, h in valid_boxes:
                    lf.write(f"{v5_cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_stats[split][v5_cid] += 1

            split_manifest[split].append({
                "image": name,
                "label": target_lbl_p.name,
                "instances": len(valid_boxes)
            })

    # Save dataset.yaml
    dataset_yaml_content = f"""# EcoVision AI V5 Dataset Configuration
# 7-Class Disposal-Oriented Waste Segregation
# Aligned with MoEFCC/CPCB Solid Waste Management Rules 2026

path: {V5_DIR.as_posix()}
train: images/train
val: images/val

nc: 7
names: {CLASS_NAMES_V5}
"""
    yaml_path = V5_DIR / "dataset.yaml"
    yaml_path.write_text(dataset_yaml_content, encoding="utf-8")
    print(f"\nWrote {yaml_path}")

    # Also save a copy at data/dataset_v5.yaml
    (WORKSPACE / "data" / "dataset_v5.yaml").write_text(dataset_yaml_content, encoding="utf-8")
    print(f"Wrote {WORKSPACE / 'data' / 'dataset_v5.yaml'}")

    # Save manifest
    manifest_path = V5_DIR / "dataset_manifest.json"
    manifest_data = {
        "version": "5.0",
        "created_at": "2026-10-31",
        "taxonomy": CLASS_NAMES_V5,
        "counts": {
            "train_images": len(split_manifest["train"]),
            "val_images": len(split_manifest["val"]),
            "total_images": len(split_manifest["train"]) + len(split_manifest["val"]),
            "train_instances": sum(class_stats["train"].values()),
            "val_instances": sum(class_stats["val"].values()),
            "total_instances": sum(class_stats["train"].values()) + sum(class_stats["val"].values()),
            "class_instances_train": {CLASS_NAMES_V5[i]: class_stats["train"][i] for i in range(7)},
            "class_instances_val": {CLASS_NAMES_V5[i]: class_stats["val"][i] for i in range(7)},
        }
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    # Save split manifest for experiment record
    exp_split_manifest = EXP_DIR / "split_manifest.json"
    exp_split_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Wrote {exp_split_manifest}")

    print("\n" + "=" * 80)
    print("V5 DATASET SUMMARY:")
    print(f"Train images: {manifest_data['counts']['train_images']} | Instances: {manifest_data['counts']['train_instances']}")
    print(f"Val images  : {manifest_data['counts']['val_images']} | Instances: {manifest_data['counts']['val_instances']}")
    print("Class breakdown (Train):", manifest_data['counts']['class_instances_train'])
    print("Class breakdown (Val)  :", manifest_data['counts']['class_instances_val'])
    print("=" * 80)


if __name__ == "__main__":
    main()

