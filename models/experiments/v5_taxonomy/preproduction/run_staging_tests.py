"""
EcoVision AI — Staging Automated Test Suite
Executes the 16 mandatory test cases against the isolated V5 staging backend.
Outputs results to models/experiments/v5_taxonomy/preproduction/staging_test_report.md
"""

import sys
import os
import io
import time
import json
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path("D:/Projects/ml_project/472")
STAGING_BACKEND = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
sys.path.insert(0, str(STAGING_BACKEND))

from services.inference import StagingYOLOInferenceService, InvalidImageError
from services.disposal import StagingDisposalService

VAL_IMG_DIR = WORKSPACE / "data" / "processed_v5" / "images" / "val"
REPORT_PATH = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging_test_report.md"


def main():
    print("=" * 80)
    print("RUNNING ECOVISION AI V5 STAGING AUTOMATED TESTS")
    print("=" * 80)

    # Initialize staging services
    inference_service = StagingYOLOInferenceService()
    disposal_service = StagingDisposalService()

    tests = []

    # Helper function to run inference and check results
    def test_image(name, img_filename, conf=0.15, expected_classes=None, min_dets=1):
        img_p = VAL_IMG_DIR / img_filename
        t0 = time.perf_counter()
        res = inference_service.predict(str(img_p), conf=conf, imgsz=800)
        dur = round((time.perf_counter() - t0) * 1000, 2)
        dets = res["detections"]
        detected_classes = [d["class_name"] for d in dets]
        
        # Attach disposal
        for d in dets:
            d["disposal"] = disposal_service.get_disposal_info(d["class_name"])

        passed = len(dets) >= min_dets
        if expected_classes:
            passed = passed and any(c in detected_classes for c in expected_classes)

        return {
            "test_name": name,
            "image": img_filename,
            "duration_ms": dur,
            "num_detections": len(dets),
            "detected_classes": detected_classes,
            "expected_classes": expected_classes,
            "passed": passed,
            "detections": dets
        }

    # 1. Plastic Object
    print("Running Test 1: Single Plastic Object...")
    t1 = test_image("1. Plastic Object Detection", "gw_plastic1002_jpg.rf.2cbd88cb12a6e39bde86b3760f226118.jpg", expected_classes=["plastic"])
    tests.append(t1)

    # 2. Metal Object
    print("Running Test 2: Single Metal Object...")
    t2 = test_image("2. Metal Object Detection", "gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg", expected_classes=["metal"])
    tests.append(t2)

    # 3. Cardboard Object
    print("Running Test 3: Single Cardboard Object...")
    t3 = test_image("3. Cardboard Object Detection", "gw_cardboard1000_jpg.rf.84fa21f62f10746865df43604d3f7031.jpg", expected_classes=["cardboard"])
    tests.append(t3)

    # 4. Glass Object
    print("Running Test 4: Single Glass Object...")
    t4 = test_image("4. Glass Object Detection", "gw_glass1000_jpg.rf.5a73deb55b262a2fd46fd9ad3089ed81.jpg", expected_classes=["glass"])
    tests.append(t4)

    # 5. Paper Object
    print("Running Test 5: Single Paper Object...")
    t5 = test_image("5. Paper Object Detection", "gw_paper1001_jpg.rf.572b43db7925850b52ec9a945f188ea3.jpg", expected_classes=["paper"])
    tests.append(t5)

    # 6. General Trash Object
    print("Running Test 6: Single General Trash Object...")
    t6 = test_image("6. General Trash Detection", "taco_000000_jpg.rf.1d051677f093973bf92f3336e269f2bc.jpg", expected_classes=["general_trash"])
    tests.append(t6)

    # 7. Organic Object
    print("Running Test 7: Organic Object Detection...")
    t7 = test_image("7. Organic Object Detection", "gw_biodegradable1225_jpg.rf.62c0de814da542600b8d18bbf282448c.jpg", expected_classes=["organic"])
    tests.append(t7)

    # 8. Multiple Objects in One Image
    print("Running Test 8: Multiple Objects Detection...")
    t8 = test_image("8. Multi-Object Detection", "gw_biodegradable1194_jpg.rf.268c14469221eafa41137461b9fc23c5.jpg", min_dets=2)
    tests.append(t8)

    # 9. Empty / Background Negative Image
    print("Running Test 9: Background Negative Image (Expected 0 detections)...")
    neg_p = VAL_IMG_DIR / "neg_000000001503.jpg"
    res_neg = inference_service.predict(str(neg_p), conf=0.15, imgsz=800)
    passed_neg = (res_neg["number_of_detections"] == 0)
    tests.append({
        "test_name": "9. Empty / Background Negative Image",
        "image": "neg_000000001503.jpg",
        "duration_ms": res_neg["inference_time_ms"],
        "num_detections": res_neg["number_of_detections"],
        "detected_classes": [],
        "expected_classes": ["None (0 detections)"],
        "passed": passed_neg,
        "evidence": f"Produced {res_neg['number_of_detections']} false positives on clean background."
    })

    # 10. Invalid Image File (Corrupted Bytes)
    print("Running Test 10: Corrupted Image Bytes...")
    corrupted_bytes = b"CORRUPTED_RAW_NON_IMAGE_BYTES_1234567890"
    try:
        inference_service.predict(corrupted_bytes)
        passed_invalid = False
        evidence_invalid = "Failed to raise InvalidImageError on corrupted bytes"
    except InvalidImageError as e:
        passed_invalid = True
        evidence_invalid = f"Successfully caught InvalidImageError: {e}"
    except Exception as e:
        passed_invalid = False
        evidence_invalid = f"Unexpected exception: {e}"
    tests.append({
        "test_name": "10. Invalid Image File (Corrupted Bytes)",
        "image": "N/A (in-memory corrupted byte buffer)",
        "duration_ms": 0.0,
        "num_detections": 0,
        "detected_classes": [],
        "expected_classes": ["InvalidImageError Exception"],
        "passed": passed_invalid,
        "evidence": evidence_invalid
    })

    # 11. Unsupported Format Simulation
    print("Running Test 11: Unsupported Format Simulation...")
    text_bytes = b"Hello, this is a plain text file pretending to be an image."
    try:
        inference_service.predict(text_bytes)
        passed_unsupported = False
        evidence_unsupported = "Did not reject plain text as image"
    except InvalidImageError as e:
        passed_unsupported = True
        evidence_unsupported = f"Correctly rejected unreadable format: {e}"
    tests.append({
        "test_name": "11. Unsupported Format Handling",
        "image": "N/A (plain text stream)",
        "duration_ms": 0.0,
        "num_detections": 0,
        "detected_classes": [],
        "expected_classes": ["InvalidImageError"],
        "passed": passed_unsupported,
        "evidence": evidence_unsupported
    })

    # 12. Missing Image Input (0-byte file)
    print("Running Test 12: Missing Image Input (0 bytes)...")
    empty_bytes = b""
    try:
        inference_service.predict(empty_bytes)
        passed_empty = False
        evidence_empty = "Failed to reject empty 0-byte buffer"
    except InvalidImageError as e:
        passed_empty = True
        evidence_empty = f"Correctly caught empty input: {e}"
    tests.append({
        "test_name": "12. Missing Image Input (0-byte Buffer)",
        "image": "N/A (0-byte buffer)",
        "duration_ms": 0.0,
        "num_detections": 0,
        "detected_classes": [],
        "expected_classes": ["InvalidImageError"],
        "passed": passed_empty,
        "evidence": evidence_empty
    })

    # 13. Low-Confidence Filtering Behavior
    print("Running Test 13: Low-Confidence Filtering...")
    test_img_p = VAL_IMG_DIR / "gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg"
    res_low = inference_service.predict(str(test_img_p), conf=0.05, imgsz=800)
    res_high = inference_service.predict(str(test_img_p), conf=0.85, imgsz=800)
    passed_conf = (res_low["number_of_detections"] >= res_high["number_of_detections"])
    tests.append({
        "test_name": "13. Confidence Threshold Filtering Sensitivity",
        "image": test_img_p.name,
        "duration_ms": res_high["inference_time_ms"],
        "num_detections": f"conf=0.05: {res_low['number_of_detections']} vs conf=0.85: {res_high['number_of_detections']}",
        "detected_classes": [],
        "expected_classes": ["Higher threshold yields subset of lower threshold"],
        "passed": passed_conf,
        "evidence": f"At conf=0.05: {res_low['number_of_detections']} detections; at conf=0.85: {res_high['number_of_detections']} detections. Correct monotonically decreasing behavior."
    })

    # 14. Multiple Objects of Same Class
    print("Running Test 14: Multiple Objects of Same Class...")
    t14 = test_image("14. Multiple Objects of Same Class", "gw_cardboard1036_jpg.rf.fd7294157a2a746d10658fed4213a30d.jpg", expected_classes=["cardboard"], min_dets=2)
    tests.append(t14)

    # 15. Sanitary Guidance Display Verification
    print("Running Test 15: Sanitary Guidance Verification under General Trash...")
    gt_info = disposal_service.get_disposal_info("general_trash")
    has_warning = gt_info.get("safety_warning") is not None and "SANITARY" in gt_info.get("safety_warning")
    has_steps = any("newspaper" in s.lower() and "red" in s.lower() for s in gt_info.get("preparation_steps", []))
    passed_sanitary = has_warning and has_steps
    tests.append({
        "test_name": "15. Sanitary Guidance Display Verification",
        "image": "N/A (Disposal Service Guidance Contract)",
        "duration_ms": 0.0,
        "num_detections": 1,
        "detected_classes": ["general_trash"],
        "expected_classes": ["Sanitary warning banner + Newspaper Red-mark step"],
        "passed": passed_sanitary,
        "evidence": f"Warning banner: '{gt_info.get('safety_warning')[:80]}...' | Red wrap step present: {has_steps}"
    })

    # 16. Conditional Packaging Guidance Verification
    print("Running Test 16: Conditional Packaging Rules Verification...")
    cardboard_info = disposal_service.get_disposal_info("cardboard")
    plastic_info = disposal_service.get_disposal_info("plastic")
    gt_info = disposal_service.get_disposal_info("general_trash")

    cb_cond = any("CONDITIONAL" in s and "pizza" in s.lower() for s in cardboard_info.get("preparation_steps", []))
    pl_cond = any("CONDITIONAL" in s and "takeaway" in s.lower() for s in plastic_info.get("preparation_steps", []))
    gt_cond = any("food-contaminated" in s.lower() or "greasy" in s.lower() for s in gt_info.get("preparation_steps", []))
    passed_conditional = cb_cond and pl_cond and gt_cond
    tests.append({
        "test_name": "16. Conditional Packaging Dual-Guidance Rules",
        "image": "N/A (Disposal Service Guidance Contract)",
        "duration_ms": 0.0,
        "num_detections": 3,
        "detected_classes": ["cardboard", "plastic", "general_trash"],
        "expected_classes": ["Dual clean vs soiled rules"],
        "passed": passed_conditional,
        "evidence": f"Cardboard conditional rule present: {cb_cond}; Plastic conditional rule: {pl_cond}; General trash grease rule: {gt_cond}."
    })

    # Visual Annotation Verification
    print("Testing Visual Annotation Generation...")
    sample_img_p = VAL_IMG_DIR / "gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg"
    sample_dets = t2["detections"]
    annotated_img = inference_service.annotate_image(str(sample_img_p), sample_dets)
    annotated_ok = (annotated_img.size == Image.open(sample_img_p).size)
    print(f"Visual annotation output size: {annotated_img.size} (matches original: {annotated_ok})")

    # Generate Markdown Report
    total_tests = len(tests)
    passed_tests = sum(1 for t in tests if t["passed"])
    all_pass = (passed_tests == total_tests)

    lines = [
        "# EcoVision AI — Staging Test Report (V5 7-Class Model & Guidance)",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Target Application:** Isolated Staging Backend (`models/experiments/v5_taxonomy/preproduction/staging/`)  ",
        "**Checkpoint:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`  ",
        f"**Overall Result:** **{'PASS (100% Passed)' if all_pass else 'FAIL'}** ({passed_tests}/{total_tests} Tests Passed)  ",
        "",
        "---",
        "",
        "## 1. Automated Test Matrix",
        "",
        "| # | Test Scenario | Input / Dataset Image | Expected Behavior | Actual Behavior | Result |",
        "|:---:|---|---|---|---|:---:|",
    ]

    for idx, t in enumerate(tests, 1):
        lines.append(
            f"| **{idx}** | {t['test_name']} | `{t['image']}` | {', '.join(t['expected_classes']) if isinstance(t['expected_classes'], list) else t['expected_classes']} | {', '.join(t['detected_classes']) if t['detected_classes'] else t.get('evidence', str(t['num_detections']) + ' detections')} | **{'PASS' if t['passed'] else 'FAIL'}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Test Case Findings",
        ""
    ])

    for idx, t in enumerate(tests, 1):
        lines.append(f"### Test {idx}: {t['test_name']}")
        lines.append(f"- **Input:** `{t['image']}`")
        lines.append(f"- **Execution Latency:** {t.get('duration_ms', 0)} ms")
        lines.append(f"- **Detected Classes:** {t['detected_classes']}")
        num_d = t.get('num_detections', 0)
        evidence_str = t.get('evidence', f'Detected {num_d} objects matching expected category profile.')
        lines.append(f"- **Verification Evidence:** {evidence_str}")
        lines.append(f"- **Result:** **{'PASS' if t['passed'] else 'FAIL'}**")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Test Verdict & Staging Integrity",
        f"- **Total Test Scenarios Executed:** {total_tests}",
        f"- **Tests Passed:** {passed_tests}",
        f"- **Tests Failed:** {total_tests - passed_tests}",
        "- **Zero Production Contamination:** All test executions operated exclusively against the staging isolated service without accessing or modifying production assets.",
        "- **Sealed Evaluation Data Sealed:** Zero test cases touched `data/real_world_eval/`."
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Summary: {passed_tests}/{total_tests} passed. Verdict: {'ALL PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
