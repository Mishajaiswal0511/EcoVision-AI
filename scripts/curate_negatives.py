"""
curate_negatives.py  –  EcoVision AI  –  Curate Verified Non-Waste Negative Images
===================================================================================
Curates genuine non-waste negative background images from COCO 2017:
  1. Parses official COCO 2017 dense instance annotations.
  2. Strictly EXCLUDES any image containing bottles, cups, wine glasses, bowls,
     utensils, food/fruit/vegetables, potted plants, books, or any target waste category.
  3. Targets specific real-world false-positive classes:
     - Electronics & chargers: cell phone, laptop, keyboard, mouse, remote, tv
     - Furniture & indoor scenes: chair, couch, bed, dining table, sink, refrigerator
     - Personal items & tools: toothbrush, backpack, handbag, suitcase, clock, scissors
  4. Downloads exactly 800 verified images to data/raw/negatives/images/.
  5. Generates empty (0-byte) YOLO label files in data/raw/negatives/labels/.
  6. Outputs a full audit manifest: data/raw/negatives/negatives_manifest.json.
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

EXCLUDE_CLASSES = {
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'book'
}

TARGET_NEGATIVE_CLASSES = {
    'cell phone', 'laptop', 'mouse', 'remote', 'keyboard', 'tv',
    'chair', 'couch', 'bed', 'dining table', 'toothbrush',
    'hair drier', 'scissors', 'clock', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'backpack', 'handbag', 'suitcase'
}


def curate_negative_dataset(
    zip_path: Path,
    out_dir: Path,
    target_count: int = 800,
    max_workers: int = 16
):
    print(f"\n============================================================")
    print(f"  EcoVision AI — Negative Background Image Curation")
    print(f"  Source Zip      : {zip_path}")
    print(f"  Target Negatives: {target_count} images")
    print(f"  Output Directory: {out_dir}")
    print(f"============================================================\n")

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("[1/4] Extracting & parsing COCO 2017 instance annotations...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open('annotations/instances_val2017.json') as f:
            coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in coco['categories']}
    img_to_cats = defaultdict(set)
    for ann in coco['annotations']:
        img_to_cats[ann['image_id']].add(cat_map[ann['category_id']])

    print("[2/4] Filtering candidate images against exclusion & target criteria...")
    candidates = []
    for img in coco['images']:
        iid = img['id']
        c_set = img_to_cats.get(iid, set())
        # Rule 1: STRICTLY zero waste or waste-ambiguous items
        if c_set.intersection(EXCLUDE_CLASSES):
            continue
        # Rule 2: Must contain target negative concepts (chargers, electronics, indoor furniture, hygiene)
        matched_targets = c_set.intersection(TARGET_NEGATIVE_CLASSES)
        if matched_targets:
            candidates.append((img, matched_targets))

    print(f"  Found {len(candidates)} verified candidate negative images in val2017.")

    # Select target_count images with balanced diversity across target concepts
    # Prioritize electronics/chargers, toothbrushes, furniture
    priority_order = [
        'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'tv',
        'toothbrush', 'hair drier', 'scissors', 'toaster', 'microwave',
        'chair', 'couch', 'bed', 'dining table', 'sink', 'refrigerator',
        'backpack', 'handbag', 'suitcase', 'clock'
    ]

    selected = []
    selected_ids = set()

    for prio_cat in priority_order:
        for img, cats in candidates:
            if img['id'] not in selected_ids and prio_cat in cats:
                selected.append((img, cats))
                selected_ids.add(img['id'])
                if len(selected) >= target_count:
                    break
        if len(selected) >= target_count:
            break

    # If still need more to reach target_count, take remaining candidates
    if len(selected) < target_count:
        for img, cats in candidates:
            if img['id'] not in selected_ids:
                selected.append((img, cats))
                selected_ids.add(img['id'])
                if len(selected) >= target_count:
                    break

    selected = selected[:target_count]
    print(f"  Selected exactly {len(selected)} diverse negative images.")

    # Category representation breakdown
    cat_dist = Counter()
    for _, cats in selected:
        for c in cats:
            cat_dist[c] += 1
    print("\n  Negative Image Content Distribution:")
    for c, cnt in cat_dist.most_common():
        print(f"    - {c}: {cnt}")

    print(f"\n[3/4] Downloading {len(selected)} verified images from COCO...")

    def download_image(entry):
        img_info, cats = entry
        fname = img_info['file_name']
        dest_img = out_images / fname
        dest_lbl = out_labels / f"{Path(fname).stem}.txt"
        url = f"http://images.cocodataset.org/val2017/{fname}"

        # Download if not already present
        if not dest_img.exists() or dest_img.stat().st_size == 0:
            urllib.request.urlretrieve(url, dest_img)

        # Verify image integrity
        with Image.open(dest_img) as im:
            im.verify()
            width, height = im.size

        # Create empty 0-byte label file
        if not dest_lbl.exists():
            dest_lbl.write_text("", encoding="utf-8")

        # Checksum
        with open(dest_img, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        return {
            "image_id": img_info['id'],
            "file_name": fname,
            "coco_url": url,
            "width": width,
            "height": height,
            "sha256": sha256,
            "contained_categories": sorted(list(cats)),
            "label_file": dest_lbl.name,
            "label_bytes": 0
        }

    manifest_entries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, entry): entry for entry in selected}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading negatives"):
            try:
                res = future.result()
                manifest_entries.append(res)
            except Exception as e:
                entry = futures[future]
                print(f"Error downloading {entry[0]['file_name']}: {e}")

    manifest_entries.sort(key=lambda x: x["image_id"])

    manifest_path = out_dir / "negatives_manifest.json"
    manifest = {
        "dataset_name": "EcoVision AI Verified Non-Waste Background Negatives",
        "source": "Microsoft COCO 2017 val2017",
        "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
        "total_images": len(manifest_entries),
        "total_labels": len(manifest_entries),
        "label_format": "YOLO 0-byte empty label files",
        "excluded_classes": sorted(list(EXCLUDE_CLASSES)),
        "target_categories_represented": dict(cat_dist),
        "images": manifest_entries
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[4/4] Negative Curation Complete!")
    print(f"  Successfully saved {len(manifest_entries)} images to {out_images}")
    print(f"  Successfully saved {len(manifest_entries)} empty labels to {out_labels}")
    print(f"  Audit manifest written to {manifest_path}")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Curate Verified Non-Waste Negatives")
    parser.add_argument("--zip_path", default="data/raw/annotations_trainval2017.zip")
    parser.add_argument("--out_dir", default="data/raw/negatives")
    parser.add_argument("--target_count", type=int, default=800)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    curate_negative_dataset(
        Path(args.zip_path),
        Path(args.out_dir),
        target_count=args.target_count,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()

