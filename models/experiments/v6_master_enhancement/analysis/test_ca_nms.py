"""
Containment-Aware Adaptive NMS (CA-NMS) Optimization Engine
Implements:
1. Standard Class-Agnostic NMS (IoU threshold = 0.45)
2. Hierarchical Containment Suppression (HCS):
   - Resolves aggregate cluster envelopes vs individual boxes
   - Suppresses cross-class ghost proposals deeply embedded inside dominant containers
3. Resolution sensitivity comparison (800 vs 1024 px)
"""

import sys
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np

WORKSPACE = Path("D:/Projects/ml_project/472")
PHOTOS_DIR = Path("D:/Projects/ml_project/waste-ai/my_photos")

TEST_IMAGES = {
    "Cardboard_Pile": PHOTOS_DIR / "cardboard_03.png",
    "Broken_Glass": PHOTOS_DIR / "glass_02.webp",
    "Green_Plastic": PHOTOS_DIR / "plastic_04.jpg",
    "Banana_Peel": PHOTOS_DIR / "organic_01.webp",
}

def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def box_intersection(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def apply_containment_aware_nms(boxes_dict_list, containment_thresh=0.80):
    """
    Refines detection boxes using containment geometry.
    boxes_dict_list: list of dicts with keys: 'cls', 'conf', 'xyxy', 'class_id'
    Sorted by confidence descending.
    """
    if len(boxes_dict_list) <= 1:
        return boxes_dict_list

    # Sort descending by confidence
    boxes = sorted(boxes_dict_list, key=lambda x: x["conf"], reverse=True)
    keep = [True] * len(boxes)

    # 1. Cross-Class Ghost Suppression inside high-confidence containers
    # E.g., low-conf metal (0.24) inside high-conf glass bottle (0.91)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        b_dominant = boxes[i]
        if b_dominant["conf"] >= 0.70:
            a_dominant = box_area(b_dominant["xyxy"])
            for j in range(len(boxes)):
                if i == j or not keep[j]:
                    continue
                b_candidate = boxes[j]
                # If candidate has lower confidence (< 0.40) and different class
                if b_candidate["conf"] < 0.40 and b_candidate["cls"] != b_dominant["cls"]:
                    a_candidate = box_area(b_candidate["xyxy"])
                    inter = box_intersection(b_dominant["xyxy"], b_candidate["xyxy"])
                    if a_candidate > 0:
                        cont = inter / a_candidate
                        # If candidate is >= 80% inside dominant container and much smaller
                        if cont >= containment_thresh and (a_candidate / a_dominant) < 0.20:
                            keep[j] = False

    # 2. Same-Class Duplicate Envelope Suppression
    # If a large box contains multiple smaller boxes of the same class
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        bi = boxes[i]
        ai = box_area(bi["xyxy"])
        contained_sub_boxes = []
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            bj = boxes[j]
            if bi["cls"] == bj["cls"]:
                aj = box_area(bj["xyxy"])
                if aj < ai:
                    inter = box_intersection(bi["xyxy"], bj["xyxy"])
                    if inter / aj >= containment_thresh:
                        contained_sub_boxes.append(j)

        # If the large box contains 2 or more distinct sub-boxes of the same class,
        # it is an aggregate envelope box -> suppress the redundant envelope
        if len(contained_sub_boxes) >= 2 and bi["conf"] < 0.45:
            keep[i] = False

    return [boxes[i] for i in range(len(boxes)) if keep[i]]

def run_experiment():
    print("=" * 80)
    print("TESTING CONTAINMENT-AWARE ADAPTIVE NMS (CA-NMS)")
    print("=" * 80)

    m = YOLO("models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt")

    results = {}

    for name, img_path in TEST_IMAGES.items():
        print(f"\nEvaluating {name}...")
        
        # 1. Standard Class-Agnostic NMS @ 800px (conf=0.15)
        res_800 = m.predict(img_path, conf=0.15, iou=0.45, imgsz=800, agnostic_nms=True, verbose=False)[0]
        dets_standard = []
        for b in res_800.boxes:
            dets_standard.append({
                "cls": m.names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 3),
                "xyxy": [round(float(x), 1) for x in b.xyxy[0]],
            })

        # 2. Containment-Aware NMS applied to standard detections
        dets_ca_nms = apply_containment_aware_nms(dets_standard)

        # 3. High-res (1024px) for Green Plastic
        res_1024 = m.predict(img_path, conf=0.15, iou=0.45, imgsz=1024, agnostic_nms=True, verbose=False)[0]
        dets_1024 = []
        for b in res_1024.boxes:
            dets_1024.append({
                "cls": m.names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 3),
                "xyxy": [round(float(x), 1) for x in b.xyxy[0]],
            })
        dets_1024_ca = apply_containment_aware_nms(dets_1024)

        results[name] = {
            "standard_800": {
                "count": len(dets_standard),
                "summary": [f"{d['cls']}:{d['conf']}" for d in dets_standard],
            },
            "ca_nms_800": {
                "count": len(dets_ca_nms),
                "summary": [f"{d['cls']}:{d['conf']}" for d in dets_ca_nms],
            },
            "standard_1024": {
                "count": len(dets_1024),
                "summary": [f"{d['cls']}:{d['conf']}" for d in dets_1024],
            },
            "ca_nms_1024": {
                "count": len(dets_1024_ca),
                "summary": [f"{d['cls']}:{d['conf']}" for d in dets_1024_ca],
            }
        }

        print(f"  Standard 800: {len(dets_standard)} dets -> {[d['cls']+':'+str(d['conf']) for d in dets_standard[:5]]}")
        print(f"  CA-NMS 800  : {len(dets_ca_nms)} dets -> {[d['cls']+':'+str(d['conf']) for d in dets_ca_nms[:5]]}")
        if name == "Green_Plastic":
            print(f"  High-Res 1024 CA-NMS: {len(dets_1024_ca)} dets -> {[d['cls']+':'+str(d['conf']) for d in dets_1024_ca[:5]]}")

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "analysis" / "ca_nms_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    run_experiment()

