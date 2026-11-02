"""
Detailed Verification of the Green Plastic Test Image (plastic_04.jpg)
Analyzes:
1. M4 at 800px (CA-NMS Disabled vs Enabled)
2. M4 at 1024px (CA-NMS Disabled vs Enabled)
3. Coordinates, confidence scores, IoU, and containment matrices
4. Determination of whether the boxes represent 13 real objects or duplicate/repeated predictions on one object.
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

WORKSPACE = Path("D:/Projects/ml_project/472")
img_path = Path("D:/Projects/ml_project/waste-ai/my_photos/plastic_04.jpg")

STAGING_BACKEND = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
sys.path.insert(0, str(STAGING_BACKEND))

from services.inference import StagingYOLOInferenceService

def box_iou(b1, b2):
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def box_containment(inner, outer):
    x1 = max(inner["x1"], outer["x1"])
    y1 = max(inner["y1"], outer["y1"])
    x2 = min(inner["x2"], outer["x2"])
    y2 = min(inner["y2"], outer["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a_inner = (inner["x2"] - inner["x1"]) * (inner["y2"] - inner["y1"])
    return inter / a_inner if a_inner > 0 else 0.0

def verify():
    im = Image.open(img_path)
    w, h = im.size
    print(f"Image: {img_path.name} | Dimensions: {w}x{h} px")

    service = StagingYOLOInferenceService()

    # 1. 800px CA-NMS False
    dets_800_no_ca = service.predict(img_path, conf=0.15, imgsz=800, agnostic_nms=False, containment_suppression=False)["detections"]
    # 2. 800px CA-NMS True
    dets_800_ca = service.predict(img_path, conf=0.15, imgsz=800, agnostic_nms=True, containment_suppression=True)["detections"]

    # 3. 1024px CA-NMS False
    dets_1024_no_ca = service.predict(img_path, conf=0.15, imgsz=1024, agnostic_nms=False, containment_suppression=False)["detections"]
    # 4. 1024px CA-NMS True
    dets_1024_ca = service.predict(img_path, conf=0.15, imgsz=1024, agnostic_nms=True, containment_suppression=True)["detections"]

    print(f"\n1. M4 @ 800px (CA-NMS Disabled): {len(dets_800_no_ca)} detections")
    print(f"2. M4 @ 800px (CA-NMS Enabled) : {len(dets_800_ca)} detections")
    print(f"3. M4 @ 1024px (CA-NMS Disabled): {len(dets_1024_no_ca)} detections")
    print(f"4. M4 @ 1024px (CA-NMS Enabled) : {len(dets_1024_ca)} detections")

    print("\nDetailed list of detections @ 1024px (CA-NMS Enabled):")
    for i, d in enumerate(dets_1024_ca):
        b = d["bounding_box"]
        area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        print(f"  Box {i:2d}: {d['class_name']:10} | conf: {d['confidence']:.3f} | [{b['x1']:5.1f}, {b['y1']:5.1f}, {b['x2']:5.1f}, {b['y2']:5.1f}] | area: {area:8.1f} px^2")

    # Analyze overlaps among the 1024px CA-NMS boxes
    print("\nPairwise Overlaps (IoU > 0.10 or Containment > 0.50):")
    significant_overlaps = []
    for i in range(len(dets_1024_ca)):
        for j in range(i + 1, len(dets_1024_ca)):
            b1 = dets_1024_ca[i]["bounding_box"]
            b2 = dets_1024_ca[j]["bounding_box"]
            iou = box_iou(b1, b2)
            c1_in_2 = box_containment(b1, b2)
            c2_in_1 = box_containment(b2, b1)
            if iou > 0.10 or c1_in_2 > 0.50 or c2_in_1 > 0.50:
                overlap_info = {
                    "box_pair": (i, j),
                    "box_1": f"{dets_1024_ca[i]['class_name']}:{dets_1024_ca[i]['confidence']}",
                    "box_2": f"{dets_1024_ca[j]['class_name']}:{dets_1024_ca[j]['confidence']}",
                    "iou": round(iou, 3),
                    "containment_1_in_2": round(c1_in_2, 3),
                    "containment_2_in_1": round(c2_in_1, 3),
                }
                significant_overlaps.append(overlap_info)
                print(f"  Box {i} ({overlap_info['box_1']}) vs Box {j} ({overlap_info['box_2']}): IoU={iou:.3f}, cont(i in j)={c1_in_2:.3f}, cont(j in i)={c2_in_1:.3f}")

    # Inspect the raw image visually to count real physical objects
    # What is in plastic_04.jpg?
    output = {
        "image": "plastic_04.jpg",
        "dimensions": [w, h],
        "count_800_no_ca": len(dets_800_no_ca),
        "count_800_ca": len(dets_800_ca),
        "count_1024_no_ca": len(dets_1024_no_ca),
        "count_1024_ca": len(dets_1024_ca),
        "detections_1024_ca": dets_1024_ca,
        "significant_overlaps": significant_overlaps,
    }

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "analysis" / "green_plastic_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    verify()

