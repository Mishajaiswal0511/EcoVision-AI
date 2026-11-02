"""
prepare_splits.py  –  EcoVision AI  –  Step 2
==============================================
Creates reproducible, stratified train/val/test splits from a flat
YOLO-format image+label source directory, with the following guarantees:

  1. Fixed random seed -> same split every run (reproducible).
  2. Stratified by class — each split has proportional class coverage.
  3. Near-duplicate awareness — reads a dedup_plan.json to prevent
     near-duplicate images from crossing split boundaries.
  4. Saves a split_manifest.json recording every image's assignment.
  5. NEVER overwrites an existing processed split without --force.
  6. Generates a final class-distribution comparison across splits.

Input layout (source directory):
    source/
        images/   ← all images (flat or per-class subdirs)
        labels/   ← matching .txt files

Output layout:
    data/processed/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/

Usage:
    python scripts/prepare_splits.py \
        --source    data/raw/roboflow_export \
        --out_dir   data/processed \
        --classes   data/dataset.yaml \
        --train     0.70 \
        --val       0.15 \
        --test      0.15 \
        --seed      42 \
        --dedup_plan evaluation/results/dedup_plan.json \
        --force

Requirements:
    pip install pyyaml tqdm pillow
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run: pip install pyyaml")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(yaml_path: str) -> list:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def get_primary_class(lbl_path: Path) -> int | None:
    """Return the most frequent class_id in a label file, or None if empty."""
    if not lbl_path.exists():
        return None
    counts: dict[int, int] = defaultdict(int)
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    pass
    if not counts:
        return None
    return max(counts, key=counts.get)


def collect_source_images(source_dir: Path) -> list[Path]:
    """Find all images recursively under source_dir/images/ or source_dir/."""
    img_root = source_dir / "images"
    root = img_root if img_root.exists() else source_dir
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_IMAGE_EXT
    )


def find_label(img_path: Path, source_dir: Path) -> Path:
    """Derive label path from image path relative to source_dir."""
    lbl = source_dir / "labels" / (img_path.stem + ".txt")
    if lbl.exists():
        return lbl
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        cand = Path(*parts).with_suffix(".txt")
        if cand.exists():
            return cand
    try:
        rel = img_path.relative_to(source_dir / "images")
    except ValueError:
        rel = img_path.relative_to(source_dir)
    return source_dir / "labels" / rel.with_suffix(".txt")


def load_dedup_groups(dedup_plan_path: str | None) -> list[list[str]]:
    """
    Return groups of image stem names that must not cross split boundaries.
    Each group contains stems of near-duplicate images.
    """
    if not dedup_plan_path or not Path(dedup_plan_path).exists():
        return []
    with open(dedup_plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    groups = []
    for entry in plan:
        group = [Path(entry["keeper"]).stem]
        for c in entry.get("candidates_for_removal", []):
            group.append(Path(c["path"]).stem)
        if len(group) > 1:
            groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    images: list[Path],
    source_dir: Path,
    ratios: tuple[float, float, float],
    seed: int,
    dedup_groups: list[list[str]],
) -> dict[str, list[Path]]:
    """
    Assign each image to train/val/test while:
      - Respecting class stratification.
      - Keeping near-duplicate groups in the same split.
    """
    rng = random.Random(seed)

    # Build stem -> group_id map for dedup grouping
    stem_to_group: dict[str, int] = {}
    for gid, group in enumerate(dedup_groups):
        for stem in group:
            stem_to_group[stem] = gid

    # Cluster images by their dedup group (or individual if not in any group)
    group_buckets: dict[int, list[Path]] = defaultdict(list)
    next_singleton_id = len(dedup_groups)
    for img in images:
        gid = stem_to_group.get(img.stem)
        if gid is None:
            group_buckets[next_singleton_id] = [img]
            next_singleton_id += 1
        else:
            group_buckets[gid].append(img)

    # For stratification, assign each bucket a primary class
    bucket_class: dict[int, int] = {}
    for gid, imgs in group_buckets.items():
        lbl = find_label(imgs[0], source_dir)
        cls = get_primary_class(lbl)
        bucket_class[gid] = cls if cls is not None else -1  # -1 = background

    # Group buckets by class
    class_buckets: dict[int, list[int]] = defaultdict(list)
    for gid, cls in bucket_class.items():
        class_buckets[cls].append(gid)

    train_ratio, val_ratio, _ = ratios

    train_imgs, val_imgs, test_imgs = [], [], []

    for cls, bucket_ids in class_buckets.items():
        rng.shuffle(bucket_ids)
        n = len(bucket_ids)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(0, round(n * val_ratio))
        # Ensure at least 1 image in val and test if class has ≥ 3 images
        if n >= 3:
            n_val  = max(1, n_val)
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val  = n - n_train

        for bid in bucket_ids[:n_train]:
            train_imgs.extend(group_buckets[bid])
        for bid in bucket_ids[n_train:n_train + n_val]:
            val_imgs.extend(group_buckets[bid])
        for bid in bucket_ids[n_train + n_val:]:
            test_imgs.extend(group_buckets[bid])

    return {"train": train_imgs, "val": val_imgs, "test": test_imgs}


# ---------------------------------------------------------------------------
# File copy
# ---------------------------------------------------------------------------

def copy_split(
    split: str,
    images: list[Path],
    source_dir: Path,
    out_dir: Path,
) -> list[dict]:
    """Copy image + label files to out_dir/images/{split}/ and labels/{split}/."""
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for img_path in tqdm(images, desc=f"  Copying {split}", unit="img"):
        lbl_path = find_label(img_path, source_dir)

        dest_img = img_out / img_path.name
        dest_lbl = lbl_out / img_path.with_suffix(".txt").name

        shutil.copy2(img_path, dest_img)
        has_label = False
        if lbl_path.exists():
            shutil.copy2(lbl_path, dest_lbl)
            has_label = True

        manifest_entries.append({
            "split": split,
            "image": img_path.name,
            "label": lbl_path.name if has_label else None,
            "source_image": str(img_path),
            "source_label": str(lbl_path) if has_label else None,
        })

    return manifest_entries


# ---------------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------------

def class_distribution(
    split: str,
    images: list[Path],
    source_dir: Path,
    class_names: list,
) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for img_path in images:
        lbl_path = find_label(img_path, source_dir)
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(class_names):
                                counts[class_names[cls_id]] += 1
                        except ValueError:
                            pass
    return dict(counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create reproducible train/val/test splits")
    parser.add_argument("--source",     required=True,
                        help="Source directory with images/ and labels/")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory (data/processed)")
    parser.add_argument("--classes",    required=True,
                        help="dataset.yaml with class names")
    parser.add_argument("--train",      type=float, default=0.70)
    parser.add_argument("--val",        type=float, default=0.15)
    parser.add_argument("--test",       type=float, default=0.15)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dedup_plan", default=None,
                        help="Path to dedup_plan.json (optional but recommended)")
    parser.add_argument("--from_manifest", default=None,
                        help="Path to an existing split_manifest.json to strictly preserve image assignments")
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing processed split directories")
    args = parser.parse_args()

    # Validate ratios
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.01:
        sys.exit(f"ERROR: Split ratios must sum to 1.0, got {total:.3f}")

    source_dir = Path(args.source)
    out_dir    = Path(args.out_dir)
    class_names = load_class_names(args.classes)

    # Safety guard
    if not args.force:
        existing = [(out_dir / "images" / s).exists() for s in ["train", "val", "test"]]
        if any(existing):
            sys.exit(
                "ERROR: Processed split directories already exist. "
                "Pass --force to overwrite, or verify you are not about to "
                "regenerate splits that training has already started on."
            )

    print(f"\n{'='*60}")
    print(f"  EcoVision AI — Prepare Splits")
    print(f"  Source : {source_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Seed   : {args.seed}")
    print(f"  Ratios : train={args.train} val={args.val} test={args.test}")
    print(f"{'='*60}\n")

    print("[1/5] Collecting source images...")
    images = collect_source_images(source_dir)
    print(f"      Found {len(images)} images.")
    if len(images) == 0:
        sys.exit("ERROR: No images found. Check --source path.")

    if args.from_manifest:
        print(f"[2/5] Preserving splits from manifest: {args.from_manifest}...")
        with open(args.from_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        splits = {"train": [], "val": [], "test": []}
        img_lookup = {img_p.name: img_p for img_p in images}
        for entry in manifest_data.get("entries", []):
            s = entry["split"]
            img_name = Path(entry["image"]).name
            if img_name in img_lookup:
                splits[s].append(img_lookup[img_name])
            else:
                print(f"  [WARN] Image {img_name} in manifest not found in source images")
        print(f"[3/5] Applied existing manifest assignments.")
    else:
        print("[2/5] Loading deduplication plan...")
        dedup_groups = load_dedup_groups(args.dedup_plan)
        print(f"      Loaded {len(dedup_groups)} near-duplicate groups.")

        print("[3/5] Creating stratified split...")
        splits = stratified_split(
            images, source_dir,
            (args.train, args.val, args.test),
            args.seed, dedup_groups,
        )
    for split, imgs in splits.items():
        print(f"      {split:<8}: {len(imgs)} images")

    print("[4/5] Copying files...")
    all_manifest = []
    for split, imgs in splits.items():
        entries = copy_split(split, imgs, source_dir, out_dir)
        all_manifest.extend(entries)

    print("[5/5] Saving manifests and reports...")
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "seed": args.seed,
            "ratios": {"train": args.train, "val": args.val, "test": args.test},
            "source_dir": str(source_dir.resolve()),
            "entries": all_manifest,
        }, f, indent=2)
    print(f"      Split manifest -> {manifest_path}")

    # Distribution table
    dist_report = {"generated_at": datetime.now().isoformat(), "splits": {}}
    print("\n  Class distribution across splits:")
    print(f"  {'Class':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")

    for split, imgs in splits.items():
        dist = class_distribution(split, imgs, source_dir, class_names)
        dist_report["splits"][split] = dist

    for cls in class_names:
        tr = dist_report["splits"].get("train", {}).get(cls, 0)
        vl = dist_report["splits"].get("val",   {}).get(cls, 0)
        ts = dist_report["splits"].get("test",  {}).get(cls, 0)
        print(f"  {cls:<22} {tr:>8} {vl:>8} {ts:>8}")

    dist_path = out_dir / "class_distribution.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(dist_report, f, indent=2)
    print(f"\n  Distribution report -> {dist_path}")

    print(f"\n{'='*60}")
    print(f"  Splits prepared successfully.")
    print(f"  IMPORTANT: The test split is now SEALED.")
    print(f"  Do not use test images for training or threshold tuning.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

