"""
Evaluation script to analyze YOLO11s detection behavior at thresholds:
0.10, 0.15, 0.25, 0.35 across development images.
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from ultralytics import YOLO

MODEL_PATH = "models/weights_best.pt"
VAL_LBL_DIR = "data/processed_v3/labels/val"
VAL_IMG_DIR = "data/processed_v3/images/val"
THRESHOLDS = [0.10, 0.15, 0.25, 0.35]

CLASS_NAMES = {
    0: "plastic",
    1: "metal",
    2: "cardboard",
    3: "glass",
    4: "paper",
    5: "general_trash",
}

def load_samples():
    # 1. Select 4 distinct images per class (24 positive images total)
    selected_pos = []
    seen_paths = set()
    
    for c_id in range(6):
        count = 0
        for lbl_path in sorted(glob.glob(os.path.join(VAL_LBL_DIR, "*.txt"))):
            fname = os.path.basename(lbl_path)
            if fname.startswith("neg_"):
                continue
            img_name = os.path.splitext(fname)[0] + ".jpg"
            img_path = os.path.join(VAL_IMG_DIR, img_name)
            if not os.path.exists(img_path) or img_path in seen_paths:
                continue
            with open(lbl_path, "r") as f:
                classes = list(set([int(l.split()[0]) for l in f if l.strip()]))
            if c_id in classes:
                selected_pos.append({
                    "path": img_path,
                    "filename": img_name,
                    "target_class": c_id,
                    "target_class_name": CLASS_NAMES[c_id],
                    "gt_classes": classes,
                    "gt_class_names": [CLASS_NAMES[c] for c in classes],
                    "is_negative": False,
                })
                seen_paths.add(img_path)
                count += 1
                if count >= 4:
                    break

    # 2. Select 10 negative non-waste images
    selected_neg = []
    for img_path in sorted(glob.glob(os.path.join(VAL_IMG_DIR, "neg_*.jpg")))[:10]:
        img_name = os.path.basename(img_path)
        selected_neg.append({
            "path": img_path,
            "filename": img_name,
            "target_class": -1,
            "target_class_name": "none",
            "gt_classes": [],
            "gt_class_names": [],
            "is_negative": True,
        })

    return selected_pos, selected_neg

def run_evaluation():
    pos_samples, neg_samples = load_samples()
    all_samples = pos_samples + neg_samples
    print(f"Loaded {len(pos_samples)} positive images and {len(neg_samples)} negative images (Total: {len(all_samples)}).")

    model = YOLO(MODEL_PATH)

    results_by_thresh: Dict[float, Dict[str, Any]] = {
        th: {
            "threshold": th,
            "total_pos": len(pos_samples),
            "pos_detected_count": 0,
            "pos_missed_count": 0,
            "total_pos_detections": 0,
            "pos_correct_match_count": 0,
            "pos_mismatch_count": 0,
            "total_neg": len(neg_samples),
            "neg_clean_count": 0,  # 0 detections (true negative)
            "neg_fp_image_count": 0,  # >= 1 detection (false positive image)
            "total_neg_fp_boxes": 0,
            "image_records": [],
        }
        for th in THRESHOLDS
    }

    for sample in all_samples:
        img_path = sample["path"]
        is_neg = sample["is_negative"]
        gt_classes = sample["gt_classes"]
        gt_names = sample["gt_class_names"]

        for th in THRESHOLDS:
            res = model.predict(source=img_path, conf=th, iou=0.45, imgsz=640, device="cpu", verbose=False)[0]
            boxes = res.boxes if (res.boxes is not None and len(res.boxes) > 0) else []
            num_det = len(boxes)

            pred_records = []
            matched_gt = False
            top_conf = 0.0

            if num_det > 0:
                top_conf = float(boxes.conf.max().cpu().numpy())
                for b in boxes:
                    cls_id = int(b.cls[0].cpu().numpy())
                    c_val = float(b.conf[0].cpu().numpy())
                    is_correct_cls = (cls_id in gt_classes) if not is_neg else False
                    if is_correct_cls:
                        matched_gt = True
                    pred_records.append({
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(c_val, 4),
                        "is_correct_class": is_correct_cls,
                    })

            record = {
                "filename": sample["filename"],
                "is_negative": is_neg,
                "gt_class_names": gt_names,
                "threshold": th,
                "num_detections": num_det,
                "top_confidence": round(top_conf, 4),
                "predictions": pred_records,
                "status": "",
            }

            stats = results_by_thresh[th]

            if not is_neg:
                stats["total_pos_detections"] += num_det
                if num_det == 0:
                    stats["pos_missed_count"] += 1
                    record["status"] = "MISSED (FN)"
                else:
                    stats["pos_detected_count"] += 1
                    if matched_gt:
                        stats["pos_correct_match_count"] += 1
                        record["status"] = "CORRECT (TP)"
                    else:
                        stats["pos_mismatch_count"] += 1
                        record["status"] = "CLASS MISMATCH (FP)"
            else:
                stats["total_neg_fp_boxes"] += num_det
                if num_det == 0:
                    stats["neg_clean_count"] += 1
                    record["status"] = "CLEAN (TN)"
                else:
                    stats["neg_fp_image_count"] += 1
                    record["status"] = f"FALSE ALARM (FP: {num_det} boxes)"

            stats["image_records"].append(record)

    output_file = Path("docs/threshold_eval_raw.json")
    output_file.write_text(json.dumps(results_by_thresh, indent=2), encoding="utf-8")
    print(f"Results saved to {output_file}")

    # Summary table print
    print("\n" + "="*85)
    print(f"{'Threshold':<10} | {'Pos Recall':<12} | {'Pos Correct':<12} | {'Total Pos Dets':<15} | {'Neg FP Rate':<15} | {'Neg FP Boxes':<12}")
    print("="*85)
    for th in THRESHOLDS:
        st = results_by_thresh[th]
        recall = f"{st['pos_detected_count']}/{st['total_pos']} ({st['pos_detected_count']/st['total_pos']*100:.1f}%)"
        correct = f"{st['pos_correct_match_count']}/{st['total_pos']} ({st['pos_correct_match_count']/st['total_pos']*100:.1f}%)"
        fp_rate = f"{st['neg_fp_image_count']}/{st['total_neg']} ({st['neg_fp_image_count']/st['total_neg']*100:.1f}%)"
        print(f"{th:<10.2f} | {recall:<12} | {correct:<12} | {st['total_pos_detections']:<15} | {fp_rate:<15} | {st['total_neg_fp_boxes']:<12}")
    print("="*85)

if __name__ == "__main__":
    run_evaluation()

