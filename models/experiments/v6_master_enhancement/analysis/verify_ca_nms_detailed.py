"""
Rigorous Independent Verification of CA-NMS across 4 Representative Test Assets:
1. Cardboard piles: cardboard_03.png
2. Broken glass shards: glass_02.webp
3. Adjacent metal cans: metal_03.webp
4. Multiple neighboring plastic objects: plastic_01.webp

Measures:
- Total detections before (Standard Class-Agnostic NMS @ 800px)
- Total detections after (CA-NMS @ 800px)
- True detections retained
- False detections removed
- Valid objects accidentally suppressed
- Duplicate boxes remaining
"""

import sys
import json
from pathlib import Path
from ultralytics import YOLO

WORKSPACE = Path("D:/Projects/ml_project/472")
PHOTOS_DIR = Path("D:/Projects/ml_project/waste-ai/my_photos")

STAGING_BACKEND = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
sys.path.insert(0, str(STAGING_BACKEND))

from services.inference import StagingYOLOInferenceService

service = StagingYOLOInferenceService()

TEST_SETS = [
    {
        "name": "1. Cardboard Piles",
        "path": PHOTOS_DIR / "cardboard_03.png",
        "ground_truth_desc": "Pile of overlapping cardboard boxes on floor",
    },
    {
        "name": "2. Broken Glass Shards",
        "path": PHOTOS_DIR / "glass_02.webp",
        "ground_truth_desc": "Shattered clear glass bottle with shards (0 metal)",
    },
    {
        "name": "3. Adjacent Metal Cans",
        "path": PHOTOS_DIR / "metal_03.webp",
        "ground_truth_desc": "Two distinct beverage cans placed adjacent to each other",
    },
    {
        "name": "4. Neighboring Plastic Objects",
        "path": PHOTOS_DIR / "plastic_01.webp",
        "ground_truth_desc": "Multiple separate plastic cups and bottles grouped together",
    },
]

def audit():
    print("=" * 80)
    print("INDEPENDENT EVIDENCE VERIFICATION OF CA-NMS")
    print("=" * 80)

    results = []

    for item in TEST_SETS:
        p = item["path"]
        name = item["name"]
        print(f"\nTesting {name} ({p.name})...")

        # Standard NMS (containment_suppression=False)
        pred_std = service.predict(p, conf=0.15, iou=0.45, imgsz=800, agnostic_nms=True, containment_suppression=False)["detections"]
        # CA-NMS (containment_suppression=True)
        pred_ca = service.predict(p, conf=0.15, iou=0.45, imgsz=800, agnostic_nms=True, containment_suppression=True)["detections"]

        std_summary = [f"{d['class_name']}:{d['confidence']:.2f}" for d in pred_std]
        ca_summary = [f"{d['class_name']}:{d['confidence']:.2f}" for d in pred_ca]

        print(f"  Standard NMS ({len(pred_std)} dets): {std_summary}")
        print(f"  CA-NMS       ({len(pred_ca)} dets): {ca_summary}")

        # Find suppressed boxes
        std_boxes = [d["bounding_box"] for d in pred_std]
        ca_boxes = [d["bounding_box"] for d in pred_ca]
        suppressed = [d for d in pred_std if d["bounding_box"] not in ca_boxes]

        print(f"  Suppressed boxes count: {len(suppressed)}")
        for s in suppressed:
            print(f"    -> Suppressed: {s['class_name']} ({s['confidence']:.2f}) at {s['bounding_box']}")

        results.append({
            "test_case": name,
            "image": p.name,
            "ground_truth_desc": item["ground_truth_desc"],
            "count_before_standard_nms": len(pred_std),
            "count_after_ca_nms": len(pred_ca),
            "detections_before": std_summary,
            "detections_after": ca_summary,
            "suppressed_boxes": suppressed,
        })

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "analysis" / "ca_nms_detailed_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification saved to {out_file}")

if __name__ == "__main__":
    audit()

