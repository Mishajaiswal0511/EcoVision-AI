"""
audit_dataset.py  –  EcoVision AI  –  Step 2
============================================
Scans a YOLO-format dataset directory and produces:
  1. A Markdown audit report  (audit_report.md)
  2. A machine-readable JSON report  (audit_report.json)
  3. Annotation preview images with bounding boxes drawn (preview/)

Usage (local or Colab):
    python scripts/audit_dataset.py \
        --data_dir  data/processed \
        --classes   data/dataset.yaml \
        --out_dir   evaluation/results \
        --preview_n 5

Requirements:
    pip install pillow pyyaml tqdm

Nothing is deleted or overwritten automatically.
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run: pip install pyyaml")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow not found. Run: pip install pillow")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # graceful fallback
        return iterable


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_class_names(yaml_path: str) -> list[str]:
    """Read class names from a YOLO dataset.yaml file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        # YOLO sometimes stores names as {0: 'class', 1: 'class2'}
        names = [names[k] for k in sorted(names.keys())]
    return names


def get_split_dirs(data_dir: Path, split: str):
    candidates = [split]
    if split == "val":
        candidates.append("valid")
    elif split == "valid":
        candidates.append("val")

    for s in candidates:
        img_a = data_dir / "images" / s
        lbl_a = data_dir / "labels" / s
        if img_a.exists():
            return img_a, lbl_a, s

        img_b = data_dir / s / "images"
        lbl_b = data_dir / s / "labels"
        if img_b.exists():
            return img_b, lbl_b, s

    return None, None, split

def iter_split(data_dir: Path, split: str):
    """Yield (image_path, label_path) pairs for a given split."""
    img_dir, lbl_dir, _ = get_split_dirs(data_dir, split)
    if img_dir is None or not img_dir.exists():
        return

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXT:
            continue
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        yield img_path, lbl_path


def parse_label_file(lbl_path: Path, n_classes: int):
    """
    Parse a YOLO .txt label file.
    Returns: list of (class_id, cx, cy, w, h)
    Raises ValueError with a descriptive message for invalid entries.
    """
    annotations = []
    if not lbl_path.exists():
        return annotations  # label file missing -> treat as background image

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(
                f"Line {lineno}: expected at least 5 fields, got {len(parts)}: '{line}'"
            )
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            raise ValueError(f"Line {lineno}: cannot parse numbers — {e}")

        if cls_id < 0 or cls_id >= n_classes:
            raise ValueError(
                f"Line {lineno}: class_id {cls_id} out of range [0, {n_classes-1}]"
            )
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Line {lineno}: {name}={val:.4f} is not in [0, 1] (not normalised)"
                )

        annotations.append((cls_id, cx, cy, w, h))

    return annotations


def try_open_image(img_path: Path):
    """Attempt to open and verify an image. Returns (ok, error_msg)."""
    try:
        with Image.open(img_path) as im:
            im.verify()
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Preview generator
# ---------------------------------------------------------------------------

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def draw_preview(img_path: Path, annotations: list, class_names: list, out_path: Path):
    """Save a copy of the image with bounding boxes and class labels drawn."""
    with Image.open(img_path).convert("RGB") as im:
        draw = ImageDraw.Draw(im)
        iw, ih = im.size

        for cls_id, cx, cy, w, h in annotations:
            x1 = int((cx - w / 2) * iw)
            y1 = int((cy - h / 2) * ih)
            x2 = int((cx + w / 2) * iw)
            y2 = int((cy + h / 2) * ih)

            color = PALETTE[cls_id % len(PALETTE)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 22), label, fill="white", font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------

def audit(data_dir: str, classes_yaml: str, out_dir: str, preview_n: int) -> dict:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(classes_yaml)
    n_classes = len(class_names)

    splits = ["train", "val", "test"]
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_path.resolve()),
        "classes_yaml": str(Path(classes_yaml).resolve()),
        "class_names": class_names,
        "n_classes": n_classes,
        "splits": {},
        "global_summary": {
            "total_images": 0,
            "total_annotations": 0,
            "corrupt_images": 0,
            "missing_label_files": 0,
            "invalid_annotation_files": 0,
        },
    }

    preview_dir = out_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_count = 0

    for split in splits:
        pairs = list(iter_split(data_path, split))
        if not pairs:
            print(f"  [SKIP] '{split}' split — directory not found or empty.")
            continue

        print(f"\n{'='*60}")
        print(f"  Auditing split: {split}  ({len(pairs)} images found)")
        print(f"{'='*60}")

        split_data = {
            "image_count": len(pairs),
            "annotation_count": 0,
            "corrupt_images": [],
            "missing_label_files": [],
            "invalid_annotation_files": [],
            "multi_object_images": 0,
            "background_images": 0,
            "class_instance_counts": {name: 0 for name in class_names},
            "class_image_counts": {name: 0 for name in class_names},
        }

        for img_path, lbl_path in tqdm(pairs, desc=f"  {split}", unit="img"):
            # 1. Image integrity check
            ok, err = try_open_image(img_path)
            if not ok:
                split_data["corrupt_images"].append(
                    {"file": img_path.name, "error": err}
                )
                report["global_summary"]["corrupt_images"] += 1
                continue

            # 2. Label file presence
            if not lbl_path.exists():
                split_data["missing_label_files"].append(img_path.name)
                report["global_summary"]["missing_label_files"] += 1
                continue

            # 3. Annotation validity
            try:
                annotations = parse_label_file(lbl_path, n_classes)
            except ValueError as e:
                split_data["invalid_annotation_files"].append(
                    {"file": lbl_path.name, "error": str(e)}
                )
                report["global_summary"]["invalid_annotation_files"] += 1
                continue

            # 4. Stats
            if len(annotations) == 0:
                split_data["background_images"] += 1
            elif len(annotations) > 1:
                split_data["multi_object_images"] += 1

            classes_in_image = set()
            for cls_id, *_ in annotations:
                split_data["annotation_count"] += 1
                report["global_summary"]["total_annotations"] += 1
                split_data["class_instance_counts"][class_names[cls_id]] += 1
                classes_in_image.add(class_names[cls_id])

            for cls_name in classes_in_image:
                split_data["class_image_counts"][cls_name] += 1

            # 5. Preview
            if preview_count < preview_n and len(annotations) > 0:
                out_img = preview_dir / f"{split}_{img_path.stem}_preview.jpg"
                try:
                    draw_preview(img_path, annotations, class_names, out_img)
                    preview_count += 1
                except Exception as e:
                    print(f"  [WARN] Preview failed for {img_path.name}: {e}")

        report["splits"][split] = split_data
        report["global_summary"]["total_images"] += len(pairs)

        # Per-split console summary
        print(f"  Images     : {split_data['image_count']}")
        print(f"  Annotations: {split_data['annotation_count']}")
        print(f"  Corrupt    : {len(split_data['corrupt_images'])}")
        print(f"  Missing lbls: {len(split_data['missing_label_files'])}")
        print(f"  Invalid lbls: {len(split_data['invalid_annotation_files'])}")
        print(f"  Multi-obj  : {split_data['multi_object_images']}")
        print(f"  Background : {split_data['background_images']}")
        print()
        print("  Class instance distribution:")
        for cls_name, count in split_data["class_instance_counts"].items():
            bar = "#" * min(count // max(1, split_data["annotation_count"] // 40), 40)
            print(f"    {cls_name:<20} {count:>6}  {bar}")

    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(report: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] JSON report -> {out_path}")


def write_markdown_report(report: dict, out_path: Path):
    lines = [
        "# EcoVision AI — Dataset Audit Report",
        "",
        f"> Generated: {report['generated_at']}",
        f"> Data dir : `{report['data_dir']}`",
        f"> Classes  : `{report['classes_yaml']}`",
        "",
        "---",
        "",
        "## Global Summary",
        "",
        f"| Item | Count |",
        f"|---|---|",
        f"| Total images | {report['global_summary']['total_images']} |",
        f"| Total annotations | {report['global_summary']['total_annotations']} |",
        f"| Corrupt images | {report['global_summary']['corrupt_images']} |",
        f"| Missing label files | {report['global_summary']['missing_label_files']} |",
        f"| Invalid annotation files | {report['global_summary']['invalid_annotation_files']} |",
        "",
        "## Class Names",
        "",
        "| ID | Name |",
        "|---|---|",
    ]
    for i, name in enumerate(report["class_names"]):
        lines.append(f"| {i} | `{name}` |")

    lines += ["", "---", ""]

    for split, data in report["splits"].items():
        lines += [
            f"## Split: `{split}`",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Images | {data['image_count']} |",
            f"| Annotations | {data['annotation_count']} |",
            f"| Multi-object images | {data['multi_object_images']} |",
            f"| Background images (no annotation) | {data['background_images']} |",
            f"| Corrupt images | {len(data['corrupt_images'])} |",
            f"| Missing label files | {len(data['missing_label_files'])} |",
            f"| Invalid annotation files | {len(data['invalid_annotation_files'])} |",
            "",
            "### Class Instance Distribution",
            "",
            "| Class | Instances | Images containing class |",
            "|---|---|---|",
        ]
        for cls_name in report["class_names"]:
            inst = data["class_instance_counts"].get(cls_name, 0)
            imgs = data["class_image_counts"].get(cls_name, 0)
            lines.append(f"| `{cls_name}` | {inst} | {imgs} |")

        if data["corrupt_images"]:
            lines += ["", "### [WARN] Corrupt Images", ""]
            for item in data["corrupt_images"]:
                lines.append(f"- `{item['file']}`: {item['error']}")

        if data["invalid_annotation_files"]:
            lines += ["", "### [WARN] Invalid Annotation Files", ""]
            for item in data["invalid_annotation_files"]:
                lines.append(f"- `{item['file']}`: {item['error']}")

        lines.append("")

    lines += [
        "---",
        "",
        "> **Note:** This report is generated automatically. Do not delete source images "
        "based on this report alone — review flagged files manually before taking action.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown report -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit a YOLO-format dataset")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing images/ and labels/ subdirs")
    parser.add_argument("--classes", required=True,
                        help="Path to dataset.yaml with class names")
    parser.add_argument("--out_dir", default="evaluation/results",
                        help="Directory to write audit reports and previews")
    parser.add_argument("--preview_n", type=int, default=5,
                        help="Number of annotation preview images to generate per split")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  EcoVision AI — Dataset Audit")
    print("="*60)

    try:
        report = audit(args.data_dir, args.classes, args.out_dir, args.preview_n)
    except Exception:
        print("\n[ERROR] Audit failed:")
        traceback.print_exc()
        sys.exit(1)

    out_path = Path(args.out_dir)
    write_json_report(report, out_path / "audit_report.json")
    write_markdown_report(report, out_path / "audit_report.md")

    gs = report["global_summary"]
    print(f"\n{'='*60}")
    print("  AUDIT COMPLETE")
    print(f"  Total images     : {gs['total_images']}")
    print(f"  Total annotations: {gs['total_annotations']}")
    print(f"  Issues found     : {gs['corrupt_images'] + gs['missing_label_files'] + gs['invalid_annotation_files']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

