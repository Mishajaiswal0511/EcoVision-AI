"""
Generates docs/confidence_threshold_analysis.md from docs/threshold_eval_raw.json.
"""

import json
from pathlib import Path

raw_data = json.loads(Path("docs/threshold_eval_raw.json").read_text(encoding="utf-8"))

markdown_lines = []
markdown_lines.append("# EcoVision AI — Confidence Threshold Analysis & Evaluation Report\n")
markdown_lines.append("**Date:** October 30, 2026  ")
markdown_lines.append("**Model:** YOLO11s (`models/weights_best.pt`)  ")
markdown_lines.append("**Evaluated Thresholds:** `0.10` (10%), `0.15` (15%), `0.25` (25% default), `0.35` (35%)  ")
markdown_lines.append("**Test Split:** Development Validation Set (24 positive waste images across all 6 classes + 10 verified negative non-waste images; sealed `data/real_world_eval/` was strictly excluded).  \n")
markdown_lines.append("---\n")

# 1. Executive Summary Table
markdown_lines.append("## 1. Summary Comparison Across All Four Thresholds\n")
markdown_lines.append("| Metric | `conf = 0.10` | `conf = 0.15` | `conf = 0.25` (Default) | `conf = 0.35` |")
markdown_lines.append("| :--- | :---: | :---: | :---: | :---: |")

stats = {float(k): v for k, v in raw_data.items()}

# Row 1: Positive Detection Rate (Recall)
row_recall = "| **Positive Waste Recall** (Images with >= 1 detection) | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    pct = (st["pos_detected_count"] / st["total_pos"]) * 100
    row_recall += f"**{st['pos_detected_count']}/{st['total_pos']} ({pct:.1f}%)** | "
markdown_lines.append(row_recall)

# Row 2: Positive Images Missed (Zero detections)
row_miss = "| **Missed Positive Images** (0 detections) | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    pct = (st["pos_missed_count"] / st["total_pos"]) * 100
    row_miss += f"{st['pos_missed_count']}/{st['total_pos']} ({pct:.1f}%) | "
markdown_lines.append(row_miss)

# Row 3: Correct Class Match Rate
row_match = "| **Ground-Truth Class Match Rate** (>= 1 box matches GT) | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    pct = (st["pos_correct_match_count"] / st["total_pos"]) * 100
    row_match += f"{st['pos_correct_match_count']}/{st['total_pos']} ({pct:.1f}%) | "
markdown_lines.append(row_match)

# Row 4: Total Bounding Boxes on Positive Images
row_pboxes = "| **Total Positive Detections (Boxes)** | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    row_pboxes += f"{st['total_pos_detections']} boxes | "
markdown_lines.append(row_pboxes)

# Row 5: Negative Image False Alarm Rate
row_nrate = "| **Negative Image False Alarm Rate** (Images with >= 1 FP) | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    pct = (st["neg_fp_image_count"] / st["total_neg"]) * 100
    row_nrate += f"{st['neg_fp_image_count']}/{st['total_neg']} ({pct:.1f}%) | "
markdown_lines.append(row_nrate)

# Row 6: Total False Positive Boxes on Negatives
row_nboxes = "| **Total False Positive Boxes on Non-Waste** | "
for th in [0.10, 0.15, 0.25, 0.35]:
    st = stats[th]
    row_nboxes += f"**{st['total_neg_fp_boxes']} boxes** | "
markdown_lines.append(row_nboxes)

markdown_lines.append("\n---\n")

# 2. Detailed Image-by-Image Results Table
markdown_lines.append("## 2. Image-by-Image Detection Breakdown\n")
markdown_lines.append("Detailed records for all 34 evaluated development images across the four thresholds:\n")
markdown_lines.append("| Image Filename | Target / GT Class | `conf = 0.10` | `conf = 0.15` | `conf = 0.25` | `conf = 0.35` |")
markdown_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

# Collect records by filename
images_ordered = [r["filename"] for r in stats[0.10]["image_records"]]

for fname in images_ordered:
    r010 = next(r for r in stats[0.10]["image_records"] if r["filename"] == fname)
    r015 = next(r for r in stats[0.15]["image_records"] if r["filename"] == fname)
    r025 = next(r for r in stats[0.25]["image_records"] if r["filename"] == fname)
    r035 = next(r for r in stats[0.35]["image_records"] if r["filename"] == fname)

    is_neg = r010["is_negative"]
    gt_label = "*(Negative / Clean)*" if is_neg else ", ".join(r010["gt_class_names"])

    def format_cell(r):
        num = r["num_detections"]
        if num == 0:
            return "0 det *(Missed)*" if not r["is_negative"] else "0 det *(Clean)*"
        top_c = r["top_confidence"]
        top_pred = r["predictions"][0]["class_name"] if r["predictions"] else ""
        correct = r["predictions"][0]["is_correct_class"] if r["predictions"] else False
        tag = "✓" if (correct and not r["is_negative"]) else ("✗" if not r["is_negative"] else "FP")
        return f"{num} det ({top_pred} {top_c*100:.0f}%) {tag}"

    c10 = format_cell(r010)
    c15 = format_cell(r015)
    c25 = format_cell(r025)
    c35 = format_cell(r035)

    short_name = f"`{fname[:24]}...`" if len(fname) > 27 else f"`{fname}`"
    markdown_lines.append(f"| {short_name} | {gt_label} | {c10} | {c15} | {c25} | {c35} |")

markdown_lines.append("\n*Legend: `✓` = Correct GT match; `✗` = Class mismatch or false positive on positive image; `FP` = False alarm on non-waste image.*\n")
markdown_lines.append("---\n")

# 3. Behavioral Analysis of Each Threshold
markdown_lines.append("## 3. In-Depth Behavioral Analysis\n")

markdown_lines.append("### A. `conf = 0.10` (High Sensitivity)")
markdown_lines.append("- **Positive Waste Recall:** 100.0% (24/24 images detected). Zero positive images were missed.")
markdown_lines.append("- **GT Match Rate:** 75.0% (18/24 images).")
markdown_lines.append("- **Total Detections:** 169 boxes across 24 images (~7.0 boxes/image).")
markdown_lines.append("- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (2 false positive boxes total on `neg_000000461751.jpg`).")
markdown_lines.append("- **Behavioral Observation:** Maximizes object discovery on challenging waste (e.g. small plastic fragments, crushed cans, distant packaging), but generates clutter and overlapping boxes on textured surfaces.\n")

markdown_lines.append("### B. `conf = 0.15` (Balanced Sensitivity)")
markdown_lines.append("- **Positive Waste Recall:** 100.0% (24/24 images detected). Zero positive images were missed.")
markdown_lines.append("- **GT Match Rate:** 70.8% (17/24 images).")
markdown_lines.append("- **Total Detections:** 111 boxes across 24 images (~4.6 boxes/image), reducing box clutter by **34.3%** compared to 0.10.")
markdown_lines.append("- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (2 boxes total).")
markdown_lines.append("- **Behavioral Observation:** Successfully rescues low-confidence waste objects that are completely missed at default `conf=0.25`, while pruning weak, spurious background boxes.\n")

markdown_lines.append("### C. `conf = 0.25` (Default Setting)")
markdown_lines.append("- **Positive Waste Recall:** 79.2% (19/24 images detected). **20.8% of positive images (5/24) return 0 detections.**")
markdown_lines.append("- **GT Match Rate:** 62.5% (15/24 images).")
markdown_lines.append("- **Total Detections:** 57 boxes across 24 images (~2.4 boxes/image).")
markdown_lines.append("- **False Positives on Negatives:** 1 out of 10 negative images triggered false detections (1 box total).")
markdown_lines.append("- **Behavioral Observation:** Delivers clean, conservative bounding boxes with minimal false alarms, but causes user friction on difficult images where the model's confidence scores naturally peak between 0.15 and 0.24 (e.g., `taco_000000_JPG` where top confidence was 0.2321).\n")

markdown_lines.append("### D. `conf = 0.35` (High Precision / Conservative)")
markdown_lines.append("- **Positive Waste Recall:** 75.0% (18/24 images detected). **25.0% of positive images (6/24) return 0 detections.**")
markdown_lines.append("- **GT Match Rate:** 58.3% (14/24 images).")
markdown_lines.append("- **Total Detections:** 36 boxes across 24 images (~1.5 boxes/image).")
markdown_lines.append("- **False Positives on Negatives:** **0.0% (0/10 images, 0 boxes).** Completely immune to false alarms on all tested non-waste backgrounds.")
markdown_lines.append("- **Behavioral Observation:** Highly reliable when an object is detected, but prone to false negatives on occluded or partially deformed waste.\n")

markdown_lines.append("---\n")

# 4. Optimal Threshold Recommendation
markdown_lines.append("## 4. Balanced Threshold Recommendation\n")
markdown_lines.append("### Recommended Default: `conf = 0.20` — `0.25` with Interactive Slider `0.10` — `0.50`\n")
markdown_lines.append("1. **Why `conf = 0.25` remains a solid starting default:** It maintains high precision on clean images and produces concise, legible bounding boxes without overcrowding the user interface with redundant overlapping detections.")
markdown_lines.append("2. **Why the `0.10` minimum slider fix is essential:** When a user photographs waste that is crushed, dimly lit, or partially obscured, the model's confidence often falls into the `0.15`–`0.24` range. Allowing the user to slide the threshold down to `10%` or `15%` recovers **100% of the positive images** in our test sample without causing significant false alarm surges on clean non-waste backgrounds.")
markdown_lines.append("3. **Operational Recommendation:** Keep `default = 0.25` as the initial baseline. When 0 detections occur, prompt the user with: *\"No waste items detected. Try lowering the confidence slider to 10–15% or adjusting lighting.\"*\n")

markdown_lines.append("---\n")
markdown_lines.append("## 5. Compliance & Protocol Safeguards")
markdown_lines.append("- Model weights (`models/weights_best.pt`) remained unmodified.")
markdown_lines.append("- Training configurations and hyperparameter files remained unchanged.")
markdown_lines.append("- The sealed `data/real_world_eval/` benchmark was not opened, accessed, or used.")
markdown_lines.append("- No claims of improved underlying model accuracy are made; findings reflect empirical score distributions of the existing YOLO11s checkpoint on development validation data.")

Path("docs/confidence_threshold_analysis.md").write_text("\n".join(markdown_lines), encoding="utf-8")
print("Saved docs/confidence_threshold_analysis.md successfully.")

