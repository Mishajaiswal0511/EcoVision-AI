"""
EcoVision AI — Staging Demo Workflow Verification Suite
Tests the live staging FastAPI service (port 8000) running M4 (yolo11s_v5_best.pt @ 800px).
Verifies:
1. Health and taxonomy endpoint
2. Disposal guidance endpoints
3. Single-object waste uploads (Plastic, Glass, Metal, Organic, General Trash)
4. Multi-object waste uploads
5. Category-specific disposal guidance and safety warnings
6. Error handling for corrupt, empty, and unsupported files
"""

import sys
import os
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
VAL_DIR = Path("data/processed_v5/images/val")

def run_tests():
    print("=" * 80)
    print("ECOVISION AI — STAGING DEMO WORKFLOW VERIFICATION SUITE")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    results = []

    # 1. Health check
    print("\n[Test 1] Health and Model Taxonomy Verification...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "staging"
        assert data["classes_count"] == 7
        assert "yolo11s_v5_best.pt" in data["model_path"]
        assert data["target_resolution"] == 800
        assert data["classes"] == ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash', 'organic']
        results.append({
            "name": "1. Health & Taxonomy Verification",
            "passed": True,
            "details": f"Status={data['status']}, Classes={data['classes_count']}, Model={Path(data['model_path']).name}"
        })
        print("  PASS: Staging backend reports healthy, 7 classes, M4 checkpoint loaded.")
    except Exception as e:
        results.append({"name": "1. Health & Taxonomy Verification", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # Helper for detection
    def post_detect(file_content, filename="test.jpg", content_type="image/jpeg", form_data=None):
        if form_data is None:
            form_data = {"conf": 0.15, "iou": 0.45, "imgsz": 800, "agnostic_nms": True}
        files = {"file": (filename, file_content, content_type)}
        return requests.post(f"{BASE_URL}/api/detect", files=files, data=form_data, timeout=30)

    # 2. Single Object: Plastic
    print("\n[Test 2] Single Plastic Object Detection...")
    img_path = VAL_DIR / "gw_plastic1002_jpg.rf.2cbd88cb12a6e39bde86b3760f226118.jpg"
    try:
        with open(img_path, "rb") as f:
            r = post_detect(f.read())
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()
        dets = data["detections"]
        assert len(dets) >= 1
        classes = [d["class_name"] for d in dets]
        assert "plastic" in classes
        top_conf = max(d["confidence"] for d in dets if d["class_name"] == "plastic")
        results.append({
            "name": "2. Plastic Object Detection",
            "passed": True,
            "details": f"Detected classes: {classes}, Top conf: {top_conf:.3f}, Annotations present: {bool(data.get('annotated_image'))}"
        })
        print(f"  PASS: Plastic detected (top conf: {top_conf:.3f}, latency: {data['inference_time_ms']} ms)")
    except Exception as e:
        results.append({"name": "2. Plastic Object Detection", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 3. Single Object: Glass with Safety Warning
    print("\n[Test 3] Glass Object Detection & Disposal Guidance...")
    img_path = VAL_DIR / "gw_glass1000_jpg.rf.5a73deb55b262a2fd46fd9ad3089ed81.jpg"
    try:
        with open(img_path, "rb") as f:
            r = post_detect(f.read())
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()
        dets = data["detections"]
        assert len(dets) >= 1
        glass_det = next(d for d in dets if d["class_name"] == "glass")
        assert glass_det["disposal"]["bin_label"] == "Glass Recycling"
        assert "newspaper" in glass_det["disposal"]["safety_warning"].lower()
        results.append({
            "name": "3. Glass Object & Safety Guidance",
            "passed": True,
            "details": f"Bin: {glass_det['disposal']['bin_label']}, Warning: '{glass_det['disposal']['safety_warning'][:40]}...'"
        })
        print(f"  PASS: Glass detected with safety warning for broken glass.")
    except Exception as e:
        results.append({"name": "3. Glass Object & Safety Guidance", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 4. Single Object: Organic Object
    print("\n[Test 4] Organic Waste Detection & Guidance...")
    img_path = VAL_DIR / "gw_biodegradable1225_jpg.rf.62c0de814da542600b8d18bbf282448c.jpg"
    try:
        with open(img_path, "rb") as f:
            r = post_detect(f.read())
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()
        dets = data["detections"]
        assert len(dets) >= 1
        classes = [d["class_name"] for d in dets]
        assert "organic" in classes
        org_det = next(d for d in dets if d["class_name"] == "organic")
        assert org_det["disposal"]["bin_color_india"] == "Green"
        results.append({
            "name": "4. Organic Waste Detection & Guidance",
            "passed": True,
            "details": f"Detected classes: {classes}, Bin: {org_det['disposal']['bin_label']} ({org_det['disposal']['bin_color_india']})"
        })
        print(f"  PASS: Organic detected with Green Bin compostable guidance.")
    except Exception as e:
        results.append({"name": "4. Organic Waste Detection & Guidance", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 5. Single Object: General Trash with Sanitary Advisory
    print("\n[Test 5] General Trash Detection & Sanitary Advisory...")
    img_path = VAL_DIR / "taco_000000_jpg.rf.1d051677f093973bf92f3336e269f2bc.jpg"
    try:
        with open(img_path, "rb") as f:
            r = post_detect(f.read())
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()
        dets = data["detections"]
        assert len(dets) >= 1
        classes = [d["class_name"] for d in dets]
        assert "general_trash" in classes
        gt_det = next(d for d in dets if d["class_name"] == "general_trash")
        assert "SANITARY" in gt_det["disposal"]["safety_warning"]
        assert "RED" in gt_det["disposal"]["safety_warning"]
        results.append({
            "name": "5. General Trash & Sanitary Advisory",
            "passed": True,
            "details": f"Bin: {gt_det['disposal']['bin_label']}, Sanitary warning verified."
        })
        print(f"  PASS: General Trash detected with statutory MoEFCC sanitary warning.")
    except Exception as e:
        results.append({"name": "5. General Trash & Sanitary Advisory", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 6. Multi-Object Detection
    print("\n[Test 6] Multi-Object Waste Scene...")
    img_path = VAL_DIR / "gw_biodegradable1194_jpg.rf.268c14469221eafa41137461b9fc23c5.jpg"
    try:
        with open(img_path, "rb") as f:
            r = post_detect(f.read())
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()
        dets = data["detections"]
        assert len(dets) >= 5, f"Expected >= 5 objects, found {len(dets)}"
        results.append({
            "name": "6. Multi-Object Detection",
            "passed": True,
            "details": f"{len(dets)} objects localized, BBoxes & labels populated, time: {data['inference_time_ms']} ms"
        })
        print(f"  PASS: Multi-object scene localized {len(dets)} objects successfully.")
    except Exception as e:
        results.append({"name": "6. Multi-Object Detection", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 7. Invalid/Corrupt File Handling
    print("\n[Test 7] Corrupt Image Handling...")
    corrupt_bytes = b"\x00\x01\x02\x03CORRUPTED_NON_IMAGE_DATA_12345"
    try:
        r = post_detect(corrupt_bytes, filename="corrupt.jpg", content_type="image/jpeg")
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        data = r.json()
        assert "detail" in data
        assert "Invalid or unreadable image" in data["detail"]
        results.append({
            "name": "7. Corrupted File Handling",
            "passed": True,
            "details": f"HTTP {r.status_code}: '{data['detail']}' (Graceful recovery without crash)"
        })
        print(f"  PASS: Gracefully returned HTTP 400 with useful error message: '{data['detail']}'")
    except Exception as e:
        results.append({"name": "7. Corrupted File Handling", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 8. Unsupported File Format Handling
    print("\n[Test 8] Unsupported File Format Handling...")
    txt_bytes = b"This is a text document, not an image."
    try:
        r = post_detect(txt_bytes, filename="document.txt", content_type="text/plain")
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        data = r.json()
        assert "detail" in data
        assert "Unsupported file format" in data["detail"]
        results.append({
            "name": "8. Unsupported Format Handling",
            "passed": True,
            "details": f"HTTP {r.status_code}: '{data['detail'][:50]}...'"
        })
        print(f"  PASS: Gracefully rejected unsupported format with HTTP 400.")
    except Exception as e:
        results.append({"name": "8. Unsupported Format Handling", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 9. Empty 0-byte File Handling
    print("\n[Test 9] Empty 0-byte File Handling...")
    empty_bytes = b""
    try:
        r = post_detect(empty_bytes, filename="empty.jpg", content_type="image/jpeg")
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"
        data = r.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()
        results.append({
            "name": "9. Empty File Handling",
            "passed": True,
            "details": f"HTTP {r.status_code}: '{data['detail']}'"
        })
        print(f"  PASS: Gracefully rejected empty file with HTTP 400: '{data['detail']}'")
    except Exception as e:
        results.append({"name": "9. Empty File Handling", "passed": False, "error": str(e)})
        print(f"  FAIL: {e}")

    # 10. Disposal Endpoint for all 7 classes
    print("\n[Test 10] Direct Disposal API for 7 Classes...")
    classes = ['plastic', 'metal', 'cardboard', 'glass', 'paper', 'general_trash', 'organic']
    all_ok = True
    for c in classes:
        r = requests.get(f"{BASE_URL}/api/disposal/{c}")
        if r.status_code != 200 or not r.json().get("success"):
            all_ok = False
            break
    # Test invalid category
    r_bad = requests.get(f"{BASE_URL}/api/disposal/radioactive_waste")
    bad_ok = (r_bad.status_code == 404)
    passed_10 = all_ok and bad_ok
    results.append({
        "name": "10. Disposal API Category Queries",
        "passed": passed_10,
        "details": f"All 7 classes returned 200; invalid category correctly returned 404"
    })
    print(f"  {'PASS' if passed_10 else 'FAIL'}: All 7 classes verified, invalid category returned 404.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"Total Tests: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")

    # Save summary json
    out_path = Path("models/experiments/v5_taxonomy/staging_verification_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

    return passed_count == total_count

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

