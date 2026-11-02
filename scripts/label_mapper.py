"""
label_mapper.py  –  EcoVision AI  –  Polygon & Label Remapper
=============================================================
Maps source dataset class labels to the EcoVision AI target taxonomy.
Correctly converts YOLOv11 instance segmentation polygons (class x1 y1 ... xn yn)
into normalized YOLO object detection bounding boxes (class cx cy w h).

Validation guarantees:
  1. Parses class ID separately.
  2. Parses all remaining coordinates as alternating (x, y) pairs.
  3. Rejects malformed polygons with odd coordinate counts or < 3 vertices (< 6 values).
  4. Computes true (x_min, y_min, x_max, y_max) and converts to (cx, cy, w, h).
  5. Validates finite numbers and checks coordinates are within [0, 1] within tolerance (1e-4).
  6. Filters degenerate zero-width or zero-height bounding boxes.
  7. Drops annotations mapped to null (excluded categories).
  8. Preserves the exact 6-class target taxonomy (classes 0-5).

Original raw files are left 100% UNTOUCHED.
"""

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
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
COORD_TOLERANCE = 1e-4


def load_source_classes(src_yaml: str) -> list:
    with open(src_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def load_mapping(mapping_json: str) -> dict:
    with open(mapping_json, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_annotation_to_bbox(
    parts: list[str],
    coord_tolerance: float = COORD_TOLERANCE,
) -> tuple[int, float, float, float, float] | None:
    """
    Parses an annotation token list into (class_id, cx, cy, w, h).
    Handles both:
      - YOLO polygon segmentation: class x1 y1 x2 y2 ... xn yn (n >= 3, even coordinate count)
      - YOLO bounding box: class cx cy w h (exactly 4 coordinates)

    Returns (class_id, cx, cy, w, h) or None if malformed/degenerate.
    """
    if len(parts) < 5:
        return None

    try:
        cls_id = int(parts[0])
    except ValueError:
        return None

    if cls_id < 0:
        return None

    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return None

    # Validate all coordinates are finite numbers
    if not all(math.isfinite(c) for c in coords):
        return None

    n_coords = len(coords)

    # Case A: Standard bounding box (exactly 4 coordinates: cx, cy, w, h)
    if n_coords == 4:
        cx, cy, w, h = coords
        if w <= 0.0 or h <= 0.0:
            return None
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        if (
            x1 < -coord_tolerance
            or y1 < -coord_tolerance
            or x2 > 1.0 + coord_tolerance
            or y2 > 1.0 + coord_tolerance
        ):
            return None
        # Clamp to [0, 1]
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        return (cls_id, cx, cy, w, h)

    # Case B: Polygon segmentation (alternating x, y pairs; requires >= 3 points, so >= 6 coords)
    if n_coords < 6 or n_coords % 2 != 0:
        return None  # Malformed: odd coordinate count or < 3 points

    # Validate coordinate range within tolerance
    for c in coords:
        if c < -coord_tolerance or c > 1.0 + coord_tolerance:
            return None

    # Extract alternating x and y vertices, clamped to [0.0, 1.0]
    xs = [min(1.0, max(0.0, c)) for c in coords[0::2]]
    ys = [min(1.0, max(0.0, c)) for c in coords[1::2]]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min

    if w <= 0.0 or h <= 0.0:
        return None  # Zero-area degenerate box

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    return (cls_id, cx, cy, w, h)


def remap_dataset(
    in_dir: Path,
    out_dir: Path,
    src_classes: list,
    mapping_cfg: dict,
    coord_tolerance: float = COORD_TOLERANCE,
) -> dict:
    target_classes = mapping_cfg["target_classes"]
    mappings = mapping_cfg["mappings"]

    # Target class to ID
    target_to_id = {name: idx for idx, name in enumerate(target_classes)}

    # ID map
    id_map = {}
    for src_id, src_name in enumerate(src_classes):
        target_name = mappings.get(src_name)
        if target_name is None:
            id_map[src_id] = None  # Excluded
        elif target_name in target_to_id:
            id_map[src_id] = target_to_id[target_name]
        else:
            id_map[src_id] = "UNMAPPED"

    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images_found": 0,
        "images_written": 0,
        "total_source_annotations": 0,
        "annotations_mapped": 0,
        "annotations_excluded": 0,
        "malformed_or_degenerate_removed": [],
        "unmapped_annotations": 0,
        "class_counts_before": defaultdict(int),
        "class_counts_after": defaultdict(int),
        "target_classes": target_classes,
        "target_to_id": target_to_id,
    }

    # Discover images from in_dir
    image_paths = []
    for p in sorted(in_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXT:
            image_paths.append(p)

    stats["source_images_found"] = len(image_paths)

    for img_path in tqdm(image_paths, desc="Remapping labels", unit="img"):
        parts = list(img_path.parts)
        lbl_path = None
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.exists():
                lbl_path = candidate

        if lbl_path is None or not lbl_path.exists():
            matches = list(in_dir.rglob(img_path.stem + ".txt"))
            if matches:
                lbl_path = matches[0]

        new_lines = []
        if lbl_path and lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for lineno, raw in enumerate(lines, 1):
                line = raw.strip()
                if not line:
                    continue
                tokens = line.split()
                stats["total_source_annotations"] += 1

                parsed = parse_annotation_to_bbox(tokens, coord_tolerance=coord_tolerance)
                if parsed is None:
                    stats["malformed_or_degenerate_removed"].append({
                        "file": lbl_path.name,
                        "line": lineno,
                        "token_count": len(tokens),
                        "reason": "malformed polygon, odd coords, non-finite, out-of-bounds, or zero-area"
                    })
                    continue

                src_id, cx, cy, w, h = parsed
                src_name = src_classes[src_id] if src_id < len(src_classes) else f"id_{src_id}"
                stats["class_counts_before"][src_name] += 1

                target_id = id_map.get(src_id, "UNMAPPED")
                if target_id is None:
                    stats["annotations_excluded"] += 1
                    continue
                if target_id == "UNMAPPED":
                    stats["unmapped_annotations"] += 1
                    continue

                target_name = target_classes[target_id]
                new_line = f"{target_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                new_lines.append(new_line)
                stats["annotations_mapped"] += 1
                stats["class_counts_after"][target_name] += 1

        # Only copy image and write label if there is at least one valid annotation
        if new_lines:
            dest_img = out_img_dir / img_path.name
            dest_lbl = out_lbl_dir / (img_path.stem + ".txt")

            shutil.copy2(img_path, dest_img)
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            stats["images_written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Remap source annotations and convert polygons to bboxes")
    parser.add_argument("--in_dir",   required=True, help="Raw dataset directory")
    parser.add_argument("--out_dir",  required=True, help="Output remapped directory")
    parser.add_argument("--mapping",  required=True, help="Path to label_mapping.json")
    parser.add_argument("--src_yaml", required=True, help="Path to raw data.yaml")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    print(f"\n============================================================")
    print(f"  EcoVision AI — Polygon-to-BBox Label Remapper")
    print(f"  Source : {in_dir}")
    print(f"  Output : {out_dir}")
    print(f"============================================================\n")

    src_classes = load_source_classes(args.src_yaml)
    mapping_cfg = load_mapping(args.mapping)

    stats = remap_dataset(in_dir, out_dir, src_classes, mapping_cfg)

    print("\n[OK] Remapping complete.")
    print(f"  Source images scanned     : {stats['source_images_found']}")
    print(f"  Images written            : {stats['images_written']}")
    print(f"  Source annotations        : {stats['total_source_annotations']}")
    print(f"  Mapped annotations        : {stats['annotations_mapped']}")
    print(f"  Excluded annotations      : {stats['annotations_excluded']}")
    print(f"  Malformed/degenerate dropped: {len(stats['malformed_or_degenerate_removed'])}")
    print(f"  Unmapped annotations      : {stats['unmapped_annotations']}")

    print("\nTarget Class Distribution:")
    for tgt, count in sorted(stats['class_counts_after'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {tgt:<15}: {count}")

    # Write mapping log
    log_path = out_dir / "remapping_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[OK] Saved remapping log -> {log_path}")


if __name__ == "__main__":
    main()
