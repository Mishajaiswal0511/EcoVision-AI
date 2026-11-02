"""
check_annotations.py  –  EcoVision AI  –  Detection Annotation Checker
=====================================================================
Deep annotation checker for YOLO object detection datasets.

Strict detection validation rules:
  1. Exact field count: each line must have exactly 5 values (class_id cx cy w h).
  2. Class ID range check: class_id must be integer in [0, nc - 1].
  3. Finite coordinates: all values must be valid finite floats (no NaN/Inf).
  4. Positive dimensions: width > 0 and height > 0 (strictly positive).
  5. Bounding box boundaries: box edges [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
     must be fully inside [0, 1] within a documented tolerance (COORD_TOLERANCE = 1e-4).
  6. Tiny and large box warnings.
  7. Exact duplicate detection within label files.
  8. Image-label 1-to-1 pairing cross-checks.

Outputs:
  - evaluation/results/annotation_check.json  (machine-readable)
  - evaluation/results/annotation_check.md    (human-readable)
"""

import argparse
import json
import math
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
COORD_TOLERANCE     = 1e-4      # Documented floating-point boundary tolerance
TINY_BOX_THRESHOLD  = 0.0001    # w*h < this fraction of image area
LARGE_BOX_THRESHOLD = 0.95      # w*h > this fraction of image area


def load_class_names(yaml_path: str) -> list:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def check_split(data_dir: Path, split: str, n_classes: int) -> dict:
    candidates = [split]
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")

    img_dir, lbl_dir = None, None
    for s in candidates:
        img_a = data_dir / "images" / s
        lbl_a = data_dir / "labels" / s
        if img_a.exists():
            img_dir, lbl_dir = img_a, lbl_a
            break
        img_b = data_dir / s / "images"
        lbl_b = data_dir / s / "labels"
        if img_b.exists():
            img_dir, lbl_dir = img_b, lbl_b
            break

    if img_dir is None or not img_dir.exists():
        return {"skipped": True, "reason": f"Image directory not found for split '{split}' under {data_dir}"}

    image_files = {
        p.stem: p for p in img_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_IMAGE_EXT
    }
    label_files = {
        p.stem: p for p in lbl_dir.iterdir()
        if p.suffix == ".txt"
    } if lbl_dir.exists() else {}

    # Orphan checks
    imgs_without_labels = sorted(set(image_files) - set(label_files))
    labels_without_imgs = sorted(set(label_files) - set(image_files))

    issues = {
        "field_count_errors": [],
        "invalid_class_id_errors": [],
        "non_finite_errors": [],
        "degenerate_box_errors": [],
        "out_of_range_errors": [],
        "out_of_bounds_errors": [],
        "tiny_box_warnings": [],
        "large_box_warnings": [],
        "duplicate_annotation_warnings": [],
        "images_without_labels": imgs_without_labels,
        "labels_without_images": labels_without_imgs,
    }

    total_annotations = 0
    valid_annotations = 0

    for stem, lbl_path in tqdm(label_files.items(), desc=f"  {split}", unit="file"):
        seen_boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for lineno, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            total_annotations += 1
            parts = line.split()

            # 1. Exact field count: detection labels MUST contain exactly 5 values
            if len(parts) != 5:
                issues["field_count_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "content": line, "tokens_found": len(parts)}
                )
                continue

            try:
                cls_id = int(parts[0])
            except ValueError:
                issues["invalid_class_id_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "content": parts[0], "reason": "non-integer class ID"}
                )
                continue

            # 2. Class ID range check
            if cls_id < 0 or cls_id >= n_classes:
                issues["invalid_class_id_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "cls_id": cls_id, "max_valid": n_classes - 1}
                )
                continue

            # 3. Finite coordinate values
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                issues["field_count_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "content": line, "reason": "non-numeric coordinate"}
                )
                continue

            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
                issues["non_finite_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "values": [cx, cy, w, h]}
                )
                continue

            # 4. Strictly positive width and height
            if w <= 0.0 or h <= 0.0:
                issues["degenerate_box_errors"].append(
                    {"file": lbl_path.name, "line": lineno, "w": w, "h": h}
                )
                continue

            # 5. Bounding box edges fully inside image boundaries within COORD_TOLERANCE
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            if (
                x1 < -COORD_TOLERANCE
                or y1 < -COORD_TOLERANCE
                or x2 > 1.0 + COORD_TOLERANCE
                or y2 > 1.0 + COORD_TOLERANCE
            ):
                err = {
                    "file": lbl_path.name,
                    "line": lineno,
                    "box": [cx, cy, w, h],
                    "edges": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
                    "tolerance": COORD_TOLERANCE,
                }
                issues["out_of_bounds_errors"].append(err)
                issues["out_of_range_errors"].append(err)
                continue

            # 6. Tiny box warning
            area = w * h
            if area < TINY_BOX_THRESHOLD:
                issues["tiny_box_warnings"].append(
                    {"file": lbl_path.name, "line": lineno, "area": round(area, 8)}
                )

            # 7. Very large box warning
            if area > LARGE_BOX_THRESHOLD:
                issues["large_box_warnings"].append(
                    {"file": lbl_path.name, "line": lineno, "area": round(area, 4)}
                )

            # 8. Duplicate detection within file
            box_sig = (cls_id, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
            if box_sig in seen_boxes:
                issues["duplicate_annotation_warnings"].append(
                    {"file": lbl_path.name, "line": lineno, "box": box_sig}
                )
            else:
                seen_boxes.append(box_sig)

            valid_annotations += 1

    return {
        "image_count": len(image_files),
        "label_file_count": len(label_files),
        "total_annotations_scanned": total_annotations,
        "valid_annotations": valid_annotations,
        "issues": issues,
        "issue_counts": {k: len(v) for k, v in issues.items()},
    }


def generate_markdown_report(results: dict, class_names: list, out_path: Path):
    lines = [
        "# EcoVision AI — YOLO Detection Annotation Quality Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Classes ({len(class_names)}):** {', '.join(class_names)}",
        f"**Coordinate Boundary Tolerance:** `{COORD_TOLERANCE}`",
        "",
        "---",
        "",
        "## Summary by Split",
        "",
        "| Split | Images | Labels | Annotations Scanned | Valid Annotations | Field Count Errors | Invalid Class IDs | Out-of-Bounds | Degenerate (w/h<=0) | Duplicates |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for split, r in results.items():
        if r.get("skipped"):
            lines.append(f"| {split} | — | — | — | — | SKIPPED: {r.get('reason')} | | | | |")
            continue
        ic = r["issue_counts"]
        lines.append(
            f"| `{split}` | {r['image_count']} | {r['label_file_count']} | "
            f"{r['total_annotations_scanned']} | {r['valid_annotations']} | "
            f"{ic['field_count_errors']} | {ic['invalid_class_id_errors']} | "
            f"{ic['out_of_bounds_errors']} | {ic['degenerate_box_errors']} | "
            f"{ic['duplicate_annotation_warnings']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Issues by Split",
        "",
    ])

    for split, r in results.items():
        if r.get("skipped"):
            continue
        lines.append(f"### Split: `{split}`")
        issues = r["issues"]
        has_any = any(len(v) > 0 for v in issues.values())
        if not has_any:
            lines.append("No issues found in this split.\n")
            continue

        for issue_type, items in issues.items():
            if not items:
                continue
            lines.append(f"#### {issue_type.replace('_', ' ').title()} ({len(items)})")
            sample = items[:5]
            for it in sample:
                lines.append(f"- `{json.dumps(it)}`")
            if len(items) > 5:
                lines.append(f"- *... and {len(items) - 5} more*")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Markdown report written -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Strict YOLO detection annotation validator")
    parser.add_argument("--data_dir", required=True, help="Processed dataset root")
    parser.add_argument("--classes",  required=True, help="dataset.yaml path")
    parser.add_argument("--out_dir",  default="evaluation/results", help="Report output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.classes)
    n_classes = len(class_names)

    print(f"\n============================================================")
    print(f"  EcoVision AI — Strict Detection Annotation Checker")
    print(f"  Target  : {data_dir}")
    print(f"  Classes : {n_classes} ({', '.join(class_names)})")
    print(f"  Boundary Tolerance: {COORD_TOLERANCE}")
    print(f"============================================================\n")

    splits = ["train", "val", "test"]
    results = {}
    for s in splits:
        results[s] = check_split(data_dir, s, n_classes)

    json_path = out_dir / "annotation_check.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Machine-readable report -> {json_path}")

    md_path = out_dir / "annotation_check.md"
    generate_markdown_report(results, class_names, md_path)


if __name__ == "__main__":
    main()
