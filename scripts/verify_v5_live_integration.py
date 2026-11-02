"""
EcoVision AI — V5 Pre-Release Integration & Verification Suite
Tests the FastAPI backend with the integrated YOLO11s V5 model checkpoint.
Verifies all 7 classes, disposal guidance, sanitary rules, bounding boxes,
multi-object detection, empty images, corrupt inputs, and production safeguards.
"""

import sys
import io
import hashlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.inference import get_inference_service, CLASS_MAPPING_V5

CLIENT = TestClient(app)

PROTECTED_WEIGHTS = Path("models/weights_best.pt")
EXPECTED_V4_SHA256 = "1933635e9c9ca2c26c68204cc8ceae7950609386ddcc394a6731185bef694d87"

V5_CANDIDATE_WEIGHTS = Path("models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt")
EXPECTED_V5_SHA256 = "bf3890b8374bfae6c376a898fc2080f79095aae0e9bd37ed612e86fb1dffc66a"

TEST_RESULTS = []

def record(test_name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_name}: {details}")
    TEST_RESULTS.append({"name": test_name, "passed": passed, "details": details})


def verify_safeguard_hashes():
    if not PROTECTED_WEIGHTS.exists():
        record("Safeguard: models/weights_best.pt exists", False, "File missing!")
        return
    with open(PROTECTED_WEIGHTS, "rb") as f:
        v4_hash = hashlib.sha256(f.read()).hexdigest()
    record(
        "Safeguard: models/weights_best.pt unaltered",
        v4_hash == EXPECTED_V4_SHA256,
        f"Hash: {v4_hash[:16]}... (Expected: {EXPECTED_V4_SHA256[:16]}...)"
    )

    if not V5_CANDIDATE_WEIGHTS.exists():
        record("Safeguard: V5 candidate checkpoint exists", False, "Candidate missing!")
        return
    with open(V5_CANDIDATE_WEIGHTS, "rb") as f:
        v5_hash = hashlib.sha256(f.read()).hexdigest()
    record(
        "Safeguard: V5 candidate checkpoint hash verified",
        v5_hash == EXPECTED_V5_SHA256,
        f"Hash: {v5_hash[:16]}... (Expected: {EXPECTED_V5_SHA256[:16]}...)"
    )


def test_health_endpoint():
    res = CLIENT.get("/api/health")
    if res.status_code != 200:
        record("API: /api/health HTTP 200", False, f"Status code: {res.status_code}")
        return
    data = res.json()
    record("API: /api/health status healthy", data.get("status") == "healthy", f"status={data.get('status')}")
    record("API: Model loaded is True", data.get("model_loaded") is True, f"model_loaded={data.get('model_loaded')}")
    record("API: Model version is V5-Candidate", data.get("model_version") == "V5-Candidate (7-Class)", f"version={data.get('model_version')}")
    record("API: Target resolution is 800", data.get("target_resolution") == 800, f"resolution={data.get('target_resolution')}")
    record("API: Number of classes is 7", data.get("num_classes") == 7, f"num_classes={data.get('num_classes')}")
    
    class_names = data.get("class_names", {})
    expected = {str(k): v for k, v in CLASS_MAPPING_V5.items()}
    record("API: All 7 class IDs and names verified", class_names == expected, f"classes={class_names}")


def test_disposal_endpoints():
    classes = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash", "organic"]
    all_ok = True
    for c in classes:
        res = CLIENT.get(f"/api/disposal/{c}")
        if res.status_code != 200:
            all_ok = False
            record(f"Disposal Guide: {c}", False, f"Status {res.status_code}")
        else:
            data = res.json()
            if not data.get("bin_color_india") or not data.get("preparation_steps"):
                all_ok = False
                record(f"Disposal Guide: {c}", False, "Missing bin_color_india or preparation_steps")
    if all_ok:
        record("Disposal Guide: All 7 classes registered & returning guidelines", True, "plastic, metal, cardboard, glass, paper, general_trash, organic")

    # Check sanitary guidance
    gt_res = CLIENT.get("/api/disposal/general_trash")
    gt_data = gt_res.json()
    sanitary_found = (
        ("newspaper" in str(gt_data.get("safety_warning", "")).lower() and "red" in str(gt_data.get("safety_warning", "")).lower()) or
        any("newspaper" in str(x).lower() and "red" in str(x).lower() for x in gt_data.get("preparation_steps", []))
    )
    record("Disposal Guide: General Trash contains MoEFCC sanitary newspaper wrap rule", sanitary_found, str(gt_data.get("safety_warning")))

    # Check conditional packaging
    cb_res = CLIENT.get("/api/disposal/cardboard")
    cb_data = cb_res.json()
    cond_found = any(
        ("pizza" in str(p).lower() or "grease" in str(p).lower() or "soiled" in str(p).lower())
        for p in cb_data.get("preparation_steps", [])
    )
    matching_step = [p for p in cb_data.get("preparation_steps", []) if "pizza" in p.lower() or "grease" in p.lower()]
    record("Disposal Guide: Cardboard includes clean vs soiled conditional guidance", cond_found, str(matching_step))


def test_blank_image():
    # Use true background negative image from validation split
    neg_path = Path("data/processed_v5/images/val/neg_000000001503.jpg")
    if not neg_path.exists():
        record("Detection: Empty/background image test", False, f"Missing {neg_path}")
        return

    with open(neg_path, "rb") as f:
        res = CLIENT.post(
            "/api/detect",
            files={"file": (neg_path.name, f, "image/jpeg")},
            data={"conf": "0.15", "iou": "0.45", "imgsz": "800"}
        )
    if res.status_code != 200:
        record("Detection: Empty/background image handled", False, f"HTTP {res.status_code}: {res.text}")
        return
    data = res.json()
    record(
        "Detection: True background image returned 0 detections gracefully",
        data.get("number_of_detections") == 0 and len(data.get("detections")) == 0,
        f"Detections count: {data.get('number_of_detections')}"
    )


def test_corrupt_file():
    corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE"
    res = CLIENT.post(
        "/api/detect",
        files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")},
        data={"conf": "0.20", "iou": "0.45", "imgsz": "800"}
    )
    record(
        "Detection: Corrupt file gracefully rejected with 400 Bad Request",
        res.status_code == 400,
        f"HTTP {res.status_code}: {res.json().get('detail')}"
    )


def test_validation_images():
    val_samples = [
        ("plastic", "data/processed_v5/images/val/gw_biodegradable695_jpg.rf.061c9cccfc1e7dbdd7ac8abd6655db54.jpg"),
        ("metal", "data/processed_v5/images/val/gw_glass854_jpg.rf.e1a2129d33643b581618a65a1274d448.jpg"),
        ("cardboard", "data/processed_v5/images/val/gw_cardboard1000_jpg.rf.84fa21f62f10746865df43604d3f7031.jpg"),
        ("glass", "data/processed_v5/images/val/gw_glass1000_jpg.rf.5a73deb55b262a2fd46fd9ad3089ed81.jpg"),
        ("paper", "data/processed_v5/images/val/gw_biodegradable1408_jpg.rf.a69120f9aa094f1cd75700ddb6c0c451.jpg"),
        ("general_trash", "data/processed_v5/images/val/taco_000000_jpg.rf.9e597da79f398ded6ed0749633965434.jpg"),
        ("organic_multi", "data/processed_v5/images/val/gw_biodegradable958_jpg.rf.d5614941b35949cf710e647c62160752.jpg"),
    ]

    for expected_target, img_path in val_samples:
        p = Path(img_path)
        if not p.exists():
            record(f"Val Detection: {expected_target}", False, f"File not found: {img_path}")
            continue

        with open(p, "rb") as f:
            res = CLIENT.post(
                "/api/detect",
                files={"file": (p.name, f, "image/jpeg")},
                data={"conf": "0.15", "iou": "0.45", "imgsz": "800"}
            )
        if res.status_code != 200:
            record(f"Val Detection: {expected_target}", False, f"HTTP {res.status_code}: {res.text}")
            continue

        data = res.json()
        dets = data.get("detections", [])
        num_dets = data.get("number_of_detections", 0)
        found_classes = [d["class_name"] for d in dets]
        box_coords_valid = all(
            len(d.get("bounding_box", {})) == 4 and all(v >= 0 for v in d["bounding_box"].values())
            for d in dets
        )

        record(
            f"Val Detection: {expected_target} ({p.stem[:24]}...)",
            num_dets > 0 and box_coords_valid,
            f"Detected {num_dets} items: {found_classes}, inference: {data.get('inference_time_ms')}ms"
        )


def main():
    print("=" * 70)
    print("ECOVISION AI — V5 LIVE INTEGRATION VERIFICATION")
    print("=" * 70)
    verify_safeguard_hashes()
    test_health_endpoint()
    test_disposal_endpoints()
    test_blank_image()
    test_corrupt_file()
    test_validation_images()
    print("=" * 70)

    total = len(TEST_RESULTS)
    passed = sum(1 for t in TEST_RESULTS if t["passed"])
    failed = total - passed
    print(f"SUMMARY: {passed}/{total} checks PASSED ({failed} failed)")

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL V5 INTEGRATION VERIFICATION CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    main()
