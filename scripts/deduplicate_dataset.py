"""
deduplicate_dataset.py  –  EcoVision AI  –  Step 2
===================================================
Finds exact duplicates (MD5 hash) and near-duplicates (perceptual hash)
across all images in the dataset directory.

IMPORTANT: This script NEVER deletes any files automatically.
It produces a review report and a deduplication plan that must be
reviewed and approved manually before any files are removed.

Steps performed:
  1. Compute MD5 hash for every image (fast; catches byte-exact copies).
  2. Compute perceptual hash (pHash) for every image (catches rescaled
     or JPEG re-saved copies of the same visual content).
  3. Build duplicate groups with configurable Hamming distance threshold.
  4. Save: dedup_report.json, dedup_report.md, dedup_plan.json
  5. In dedup_plan.json, each group lists one "keeper" and N "candidates
     for removal" — the human must act on this list.

Usage:
    python scripts/deduplicate_dataset.py \
        --data_dir  data/processed \
        --threshold 10 \
        --out_dir   evaluation/results

    --threshold : Hamming distance for pHash (0 = exact visual match,
                  10 = very similar, 20 = somewhat similar).
                  Recommended: 8–12 for near-duplicate detection.

Requirements:
    pip install pillow imagehash tqdm
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

try:
    import imagehash
    from PIL import Image
except ImportError:
    sys.exit("Required: pip install pillow imagehash")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of_image(path: Path):
    """Returns an imagehash.ImageHash object or None on error."""
    try:
        with Image.open(path).convert("RGB") as im:
            return imagehash.phash(im)
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Gather all images
# ---------------------------------------------------------------------------

def collect_images(data_dir: Path) -> list[dict]:
    """Walk data_dir recursively, return list of {path, split} dicts."""
    entries = []
    seen_paths = set()
    splits_to_check = ["train", "val", "valid", "test", "ood_eval", "real_world_eval"]
    for split in splits_to_check:
        candidates = [
            data_dir / "images" / split,
            data_dir / split / "images",
            data_dir / split
        ]
        for split_dir in candidates:
            if split_dir.exists():
                norm_split = "val" if split == "valid" else split
                for p in sorted(split_dir.rglob("*")):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXT and str(p) not in seen_paths:
                        seen_paths.add(str(p))
                        entries.append({"path": p, "split": norm_split})
    return entries


# ---------------------------------------------------------------------------
# Main deduplication logic
# ---------------------------------------------------------------------------

def run_deduplication(data_dir: str, threshold: int, out_dir: str) -> dict:
    data_path = Path(data_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Collecting image paths...")
    entries = collect_images(data_path)
    if not entries:
        print(f"[WARN] No images found under {data_path}. Is the dataset downloaded?")
        return {}

    print(f"      Found {len(entries)} images across all splits.")

    # ---- Stage 1: MD5 exact duplicates ----
    print("\n[2/4] Computing MD5 hashes (exact duplicate detection)...")
    md5_map: dict[str, list[Path]] = defaultdict(list)
    hash_errors = []
    for entry in tqdm(entries, desc="  MD5", unit="img"):
        try:
            h = md5_of_file(entry["path"])
            md5_map[h].append(entry["path"])
        except Exception as e:
            hash_errors.append({"file": str(entry["path"]), "error": str(e)})

    exact_groups = {h: paths for h, paths in md5_map.items() if len(paths) > 1}
    exact_dup_images = sum(len(v) - 1 for v in exact_groups.values())

    print(f"      Exact duplicate groups : {len(exact_groups)}")
    print(f"      Duplicate images found : {exact_dup_images}")

    # ---- Stage 2: pHash near-duplicates ----
    print(f"\n[3/4] Computing perceptual hashes (near-duplicate detection, threshold={threshold})...")
    phash_data: list[dict] = []
    for entry in tqdm(entries, desc="  pHash", unit="img"):
        ph = phash_of_image(entry["path"])
        if ph is not None:
            phash_data.append({
                "path": entry["path"],
                "split": entry["split"],
                "phash": ph,
            })
        else:
            hash_errors.append({
                "file": str(entry["path"]), "error": "pHash failed (corrupt/unsupported)"
            })

    print(f"      Successfully hashed: {len(phash_data)} images")
    print(f"      Skipped (errors)   : {len(hash_errors)}")

    # Pairwise comparison — O(n^2), acceptable for < 20k images
    print("      Building near-duplicate groups (pairwise comparison)...")
    parent = {i: i for i in range(len(phash_data))}  # Union-Find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    n = len(phash_data)
    near_pair_count = 0
    for i, j in tqdm(combinations(range(n), 2), desc="  Pairs",
                     total=n * (n - 1) // 2, unit="pair"):
        dist = phash_data[i]["phash"] - phash_data[j]["phash"]
        if dist <= threshold:
            union(i, j)
            near_pair_count += 1

    # Group by root
    groups_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    near_dup_groups = {r: idxs for r, idxs in groups_map.items() if len(idxs) > 1}
    near_dup_images = sum(len(v) - 1 for v in near_dup_groups.values())

    print(f"      Near-duplicate pairs  : {near_pair_count}")
    print(f"      Near-duplicate groups : {len(near_dup_groups)}")
    print(f"      Redundant images      : {near_dup_images}")

    # ---- Stage 3: Build deduplication plan ----
    print("\n[4/4] Building deduplication plan...")
    dedup_plan = []
    for root_idx, member_idxs in near_dup_groups.items():
        members = [phash_data[i] for i in member_idxs]
        # Prefer keeping the image from the train split if there's a choice
        members_sorted = sorted(
            members,
            key=lambda m: (0 if m["split"] == "train" else 1, str(m["path"]))
        )
        keeper = members_sorted[0]
        candidates = members_sorted[1:]

        # Check for cross-split contamination
        splits_present = {m["split"] for m in members}
        cross_split = len(splits_present) > 1

        dedup_plan.append({
            "group_size": len(members),
            "cross_split_contamination": cross_split,
            "splits_present": sorted(splits_present),
            "keeper": str(keeper["path"]),
            "keeper_split": keeper["split"],
            "candidates_for_removal": [
                {"path": str(c["path"]), "split": c["split"]}
                for c in candidates
            ],
        })

    cross_split_count = sum(1 for g in dedup_plan if g["cross_split_contamination"])

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_path.resolve()),
        "phash_threshold": threshold,
        "total_images_scanned": len(entries),
        "hash_errors": hash_errors,
        "exact_duplicates": {
            "group_count": len(exact_groups),
            "redundant_images": exact_dup_images,
            "groups": [
                {"hash": h, "files": [str(p) for p in paths]}
                for h, paths in exact_groups.items()
            ],
        },
        "near_duplicates": {
            "group_count": len(near_dup_groups),
            "redundant_images": near_dup_images,
            "cross_split_contamination_groups": cross_split_count,
        },
        "deduplication_plan": dedup_plan,
        "action_required": (
            "MANUAL REVIEW REQUIRED. "
            "This script has NOT deleted any files. "
            "Review dedup_plan.json, verify the keeper/removal candidates, "
            "then run a separate deletion script or remove files manually."
        ),
    }

    return report


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] JSON report -> {path}")


def write_markdown(data: dict, path: Path):
    nd = data.get("near_duplicates", {})
    ed = data.get("exact_duplicates", {})
    plan = data.get("deduplication_plan", [])

    cross = nd.get("cross_split_contamination_groups", 0)
    status = "🔴 ACTION NEEDED" if (ed["redundant_images"] + nd["redundant_images"]) > 0 else "🟢 CLEAN"

    lines = [
        "# EcoVision AI — Deduplication Report",
        "",
        f"> Generated : {data.get('generated_at', 'N/A')}",
        f"> Status    : **{status}**",
        f"> pHash threshold : {data.get('phash_threshold', 'N/A')} (Hamming distance)",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Images scanned | {data.get('total_images_scanned', 0)} |",
        f"| Hash errors | {len(data.get('hash_errors', []))} |",
        f"| **Exact duplicate groups** | **{ed.get('group_count', 0)}** |",
        f"| Exact redundant images | {ed.get('redundant_images', 0)} |",
        f"| **Near-duplicate groups** | **{nd.get('group_count', 0)}** |",
        f"| Near-duplicate redundant images | {nd.get('redundant_images', 0)} |",
        f"| **Cross-split contamination groups** | **{cross}** |",
        "",
        "> [!CAUTION]",
        "> Cross-split contamination means the same (or near-identical) image",
        "> appears in both the training set and the test/validation set.",
        "> This inflates evaluation scores and must be resolved before training.",
        "",
    ]

    if plan:
        lines += [
            "## Deduplication Plan (Top 20 Groups)",
            "",
            "_Full plan in `dedup_plan.json`. Only top 20 shown here._",
            "",
        ]
        for i, group in enumerate(plan[:20], start=1):
            contamination_tag = " 🔴 CROSS-SPLIT" if group["cross_split_contamination"] else ""
            lines.append(f"### Group {i}{contamination_tag}")
            lines.append(f"- **Keeper** (`{group['keeper_split']}`): `{Path(group['keeper']).name}`")
            for c in group["candidates_for_removal"]:
                lines.append(
                    f"- Candidate for removal (`{c['split']}`): `{Path(c['path']).name}`"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "> **Important:** No files have been deleted. This report is for review only.",
        "> After manual verification, remove duplicates using your preferred method.",
        "> Prioritise removing cross-split duplicates first.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown report -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find exact and near-duplicate images")
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--threshold", type=int, default=10,
                        help="pHash Hamming distance threshold (default: 10)")
    parser.add_argument("--out_dir",   default="evaluation/results")
    args = parser.parse_args()

    report = run_deduplication(args.data_dir, args.threshold, args.out_dir)
    if not report:
        sys.exit(0)

    out_path = Path(args.out_dir)
    write_json(report, out_path / "dedup_report.json")
    write_markdown(report, out_path / "dedup_report.md")

    # Write just the plan separately for easier review
    plan_path = out_path / "dedup_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(report["deduplication_plan"], f, indent=2)
    print(f"[OK] Dedup plan   -> {plan_path}")

    cross = report["near_duplicates"]["cross_split_contamination_groups"]
    if cross > 0:
        print(f"\n[WARN] {cross} cross-split contamination group(s) detected!")
        print("       Review dedup_plan.json and resolve before creating final splits.")
    else:
        print("\n[OK] No cross-split contamination detected.")


if __name__ == "__main__":
    main()

