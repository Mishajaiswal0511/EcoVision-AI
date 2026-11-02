"""
Deep Forensic Failure Analysis across Key Failure Assets:
1. Cardboard pile: cardboard_03.png
2. Broken glass: glass_02.webp
3. Green plastic object: plastic_04.jpg
4. Banana peel: organic_01.webp
5. Garbage pile: household-recyclable-waste / garbage-heap

Analyzes raw boxes, per-class probabilities, IoUs, containment, and effect of input resolutions.
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

WORKSPACE = Path("D:/Projects/ml_project/472")
PHOTOS_DIR = Path("D:/Projects/ml_project/waste-ai/my_photos")

MODELS = {
    "M1_V4_Prod": WORKSPACE / "models" / "weights_best.pt",
    "M2_V4_11m": WORKSPACE / "models" / "experiments" / "yolo11m_best.pt",
    "M3_V5_ExpA": WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "training" / "runs" / "exp_a_yolo11s_640" / "weights" / "best.pt",
    "M4_V5_ExpB": WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "training" / "yolo11s_v5_best.pt",
}

TEST_CASES = {
    "A_Cardboard_Pile": PHOTOS_DIR / "cardboard_03.png",
    "B_Broken_Glass": PHOTOS_DIR / "glass_02.webp",
    "C_Green_Plastic": PHOTOS_DIR / "plastic_04.jpg",
    "D_Banana_Peel": PHOTOS_DIR / "organic_01.webp",
}

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def box_containment(inner, outer):
    # What fraction of inner is covered by outer
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / a_inner if a_inner > 0 else 0.0

def analyze():
    print("Loading M4 model...")
    m4 = YOLO(str(MODELS["M4_V5_ExpB"]))

    findings = {}

    for case_id, img_path in TEST_CASES.items():
        print(f"\nAnalyzing {case_id}: {img_path.name}...")
        im = Image.open(img_path)
        w, h = im.size

        # Run at conf=0.05, agnostic_nms=False to see raw predictions
        res_raw = m4.predict(img_path, conf=0.05, imgsz=800, agnostic_nms=False, verbose=False)[0]
        # Run at conf=0.15, agnostic_nms=True
        res_agnostic = m4.predict(img_path, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)[0]
        # Test resolutions: 640 vs 800 vs 1024
        res_640 = m4.predict(img_path, conf=0.15, imgsz=640, agnostic_nms=True, verbose=False)[0]
        res_1024 = m4.predict(img_path, conf=0.15, imgsz=1024, agnostic_nms=True, verbose=False)[0]

        raw_boxes = []
        for b in res_raw.boxes:
            raw_boxes.append({
                "cls": m4.names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 4),
                "xyxy": [round(float(x), 1) for x in b.xyxy[0]],
            })

        agnostic_boxes = []
        for b in res_agnostic.boxes:
            agnostic_boxes.append({
                "cls": m4.names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 4),
                "xyxy": [round(float(x), 1) for x in b.xyxy[0]],
            })

        # Pairwise IoU and containment on raw_boxes
        overlaps = []
        for i in range(len(raw_boxes)):
            for j in range(i + 1, len(raw_boxes)):
                bi = raw_boxes[i]["xyxy"]
                bj = raw_boxes[j]["xyxy"]
                iou = box_iou(bi, bj)
                cont_ij = box_containment(bi, bj)
                cont_ji = box_containment(bj, bi)
                if iou > 0.1 or cont_ij > 0.5 or cont_ji > 0.5:
                    overlaps.append({
                        "box_1": f"{raw_boxes[i]['cls']}:{raw_boxes[i]['conf']}",
                        "box_2": f"{raw_boxes[j]['cls']}:{raw_boxes[j]['conf']}",
                        "iou": round(iou, 3),
                        "containment_1_in_2": round(cont_ij, 3),
                        "containment_2_in_1": round(cont_ji, 3),
                    })

        findings[case_id] = {
            "image_size": [w, h],
            "raw_detections_at_005": len(raw_boxes),
            "detections_at_015_agnostic": len(agnostic_boxes),
            "detections_at_640": len(res_640.boxes),
            "detections_at_1024": len(res_1024.boxes),
            "agnostic_boxes": agnostic_boxes,
            "top_raw_boxes": raw_boxes[:10],
            "significant_overlaps": overlaps[:10],
        }

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "analysis" / "forensic_failure_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print(f"\nForensic analysis completed. Data saved to {out_file}")

if __name__ == "__main__":
    analyze()

