"""
Comprehensive API test suite for EcoVision AI FastAPI backend.
Uses synthetic in-memory images. Never touches data/real_world_eval/.
"""

import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app, MAX_IMAGE_SIZE_BYTES
from backend.services.disposal import get_disposal_service
from backend.services.inference import CLASS_MAPPING


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def make_synthetic_image_bytes(
    width: int = 320,
    height: int = 240,
    color: tuple = (120, 150, 180),
    image_format: str = "PNG",
) -> bytes:
    """Creates in-memory image bytes for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=image_format)
    return buf.getvalue()


# 1. Health & Readiness Tests
def test_health_endpoint_success(client: TestClient):
    """Verify /api/health returns HTTP 200 and expected fields."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "online"
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_architecture"] == "YOLO11s"
    assert data["device"] == "cpu"
    assert data["num_classes"] == 6
    assert data["version"] == "1.0.0"


def test_health_endpoint_class_names(client: TestClient):
    """Verify /api/health exposes the exact expected class taxonomy."""
    response = client.get("/api/health")
    data = response.json()
    class_names = data["class_names"]
    # Keys might be serialized as strings in JSON
    parsed_classes = {int(k): v for k, v in class_names.items()}
    assert parsed_classes == CLASS_MAPPING


# 2. Disposal Guidance Tests
@pytest.mark.parametrize("category", ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"])
def test_disposal_lookup_all_six_classes(client: TestClient, category: str):
    """Verify /api/disposal/{class_name} returns valid guidance for all 6 target classes."""
    response = client.get(f"/api/disposal/{category}")
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] == category
    assert "material" in data and data["material"] is not None
    assert "bin_color_india" in data and data["bin_color_india"] is not None
    assert "bin_label" in data and data["bin_label"] is not None
    assert isinstance(data["preparation_steps"], list)
    assert len(data["preparation_steps"]) > 0
    assert isinstance(data["disposal_locations"], list)
    assert "recyclable" in data
    assert "compositable" in data
    assert "hazardous" in data


def test_disposal_lookup_case_insensitivity(client: TestClient):
    """Verify class lookup works regardless of casing."""
    for variant in ["PLASTIC", "Plastic", "pLaStIc"]:
        res = client.get(f"/api/disposal/{variant}")
        assert res.status_code == 200
        assert res.json()["class_name"] == "plastic"


def test_disposal_lookup_unknown_category(client: TestClient):
    """Verify unknown category returns HTTP 404 with helpful error message."""
    response = client.get("/api/disposal/unobtainium_alien_material")
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "Supported categories" in response.json()["detail"]


# 3. Detection Endpoint Tests
def test_detect_valid_image(client: TestClient):
    """Verify /api/detect with a valid synthetic image returns full detection schema."""
    image_bytes = make_synthetic_image_bytes(400, 300, color=(100, 200, 150))
    files = {"file": ("test_sample.png", image_bytes, "image/png")}
    data = {"conf": 0.25, "iou": 0.45, "imgsz": 640}

    response = client.post("/api/detect", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["image_width"] == 400
    assert res["image_height"] == 300
    assert isinstance(res["inference_time_ms"], (int, float))
    assert res["inference_time_ms"] > 0
    assert isinstance(res["number_of_detections"], int)
    assert isinstance(res["detections"], list)
    assert res["annotated_image"].startswith("data:image/png;base64,")


def test_detect_empty_detections_valid_preview(client: TestClient):
    """Verify detection on a blank image returns empty detections and valid annotated image."""
    blank_bytes = make_synthetic_image_bytes(300, 300, color=(0, 0, 0))
    files = {"file": ("blank.jpg", blank_bytes, "image/jpeg")}
    data = {"conf": 0.70}

    response = client.post("/api/detect", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["number_of_detections"] == 0
    assert res["detections"] == []
    assert res["annotated_image"].startswith("data:image/png;base64,")


def test_detect_annotated_image_base64_validity(client: TestClient):
    """Verify the returned annotated image base64 is a valid decodable PNG image."""
    image_bytes = make_synthetic_image_bytes(250, 180)
    files = {"file": ("preview_test.png", image_bytes, "image/png")}

    response = client.post("/api/detect", files=files)
    assert response.status_code == 200
    data_url = response.json()["annotated_image"]

    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)

    # Decode base64 payload
    raw_b64 = data_url[len(prefix):]
    decoded_bytes = base64.b64decode(raw_b64)
    assert len(decoded_bytes) > 0

    # Ensure Pillow can read the decoded image
    decoded_image = Image.open(io.BytesIO(decoded_bytes))
    assert decoded_image.format == "PNG"
    assert decoded_image.size == (250, 180)


def test_detect_invalid_corrupted_image(client: TestClient):
    """Verify corrupted bytes return HTTP 400."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    files = {"file": ("corrupted.png", corrupted_bytes, "image/png")}

    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "Invalid or unreadable image" in response.json()["detail"]


def test_detect_unsupported_file_format(client: TestClient):
    """Verify non-image formats (.txt, .pdf) are rejected with HTTP 400."""
    text_content = b"This is a text file, not an image."
    files = {"file": ("document.txt", text_content, "text/plain")}

    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_detect_empty_file_rejected(client: TestClient):
    """Verify zero-byte upload is rejected with HTTP 400."""
    files = {"file": ("empty.png", b"", "image/png")}

    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_oversized_image_rejected(client: TestClient, monkeypatch):
    """Verify image exceeding maximum allowed size is rejected with HTTP 413."""
    import backend.main
    # Set limit temporarily to 1000 bytes for test
    monkeypatch.setattr(backend.main, "MAX_IMAGE_SIZE_BYTES", 1000)

    big_image_bytes = make_synthetic_image_bytes(500, 500)
    assert len(big_image_bytes) > 1000

    files = {"file": ("large_image.png", big_image_bytes, "image/png")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 413
    assert "exceeds the maximum allowed limit" in response.json()["detail"]


def test_detect_invalid_parameters(client: TestClient):
    """Verify invalid conf, iou, and imgsz parameters return HTTP 400."""
    image_bytes = make_synthetic_image_bytes(100, 100)

    # Invalid conf (> 1.0)
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_conf_high = client.post("/api/detect", files=files, data={"conf": 1.5})
    assert res_conf_high.status_code == 400

    # Invalid conf (<= 0.0)
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_conf_low = client.post("/api/detect", files=files, data={"conf": -0.1})
    assert res_conf_low.status_code == 400

    # Invalid iou (> 1.0)
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_iou = client.post("/api/detect", files=files, data={"iou": 2.0})
    assert res_iou.status_code == 400

    # Invalid imgsz (<= 0)
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_imgsz = client.post("/api/detect", files=files, data={"imgsz": 0})
    assert res_imgsz.status_code == 400


def test_detect_agnostic_nms_parameter(client: TestClient):
    """Verify agnostic_nms flag is accepted and processed."""
    image_bytes = make_synthetic_image_bytes(200, 200)

    # agnostic_nms = true
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_true = client.post("/api/detect", files=files, data={"agnostic_nms": True})
    assert res_true.status_code == 200

    # agnostic_nms = false
    files = {"file": ("test.png", image_bytes, "image/png")}
    res_false = client.post("/api/detect", files=files, data={"agnostic_nms": False})
    assert res_false.status_code == 200


def test_detect_omitted_conf_defaults_to_015(client: TestClient):
    """Verify omitting the conf parameter defaults cleanly to 0.15."""
    import inspect
    from backend.main import detect_waste
    sig = inspect.signature(detect_waste)
    conf_param = sig.parameters["conf"]
    # Verify default in FastAPI Form is 0.15
    assert conf_param.default.default == 0.15

    # Run actual detection without providing 'conf' in data
    image_bytes = make_synthetic_image_bytes(200, 200)
    files = {"file": ("test_default.png", image_bytes, "image/png")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 200
    assert response.json()["success"] is True

