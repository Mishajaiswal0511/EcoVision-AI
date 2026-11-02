"""
Unit tests for EcoVision AI YOLO11s standalone inference service.
"""

import io
import pytest
import numpy as np
from PIL import Image

from backend.services.inference import (
    YOLOInferenceService,
    get_inference_service,
    CLASS_MAPPING,
    CLASS_COLORS,
    InvalidImageError,
    InferenceError,
)

MODEL_PATH = "models/weights_best.pt"


@pytest.fixture(scope="session")
def service() -> YOLOInferenceService:
    """Fixture providing the singleton inference service."""
    return get_inference_service(model_path=MODEL_PATH, device="cpu")


def test_model_loads_successfully(service):
    """Verify model loads successfully on CPU."""
    assert service is not None
    assert service.model is not None
    assert str(service.device) == "cpu"


def test_singleton_pattern():
    """Verify singleton pattern returns the same instance."""
    svc1 = get_inference_service(model_path=MODEL_PATH)
    svc2 = get_inference_service(model_path=MODEL_PATH)
    svc3 = YOLOInferenceService(model_path=MODEL_PATH)
    assert svc1 is svc2
    assert svc1 is svc3


def test_taxonomy_verification(service):
    """Verify exact 6 classes and exact class names and order."""
    expected = {
        0: "plastic",
        1: "metal",
        2: "cardboard",
        3: "glass",
        4: "paper",
        5: "general_trash",
    }
    assert CLASS_MAPPING == expected
    assert len(service.model.names) == 6

    for cid, name in expected.items():
        assert service.model.names[cid] == name


def test_valid_image_inference_structure(service):
    """Verify inference produces the expected structured dictionary schema."""
    # Create a synthetic 640x480 RGB test image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))

    res = service.predict(img, conf=0.15, iou=0.45, imgsz=640)

    assert isinstance(res, dict)
    assert res["success"] is True
    assert res["image_width"] == 640
    assert res["image_height"] == 480
    assert isinstance(res["inference_time_ms"], (int, float))
    assert res["inference_time_ms"] > 0
    assert isinstance(res["number_of_detections"], int)
    assert isinstance(res["detections"], list)
    assert res["number_of_detections"] == len(res["detections"])


def test_default_confidence_parameter_is_015(service):
    """Verify predict defaults to conf=0.15 when conf parameter is omitted."""
    import inspect
    sig = inspect.signature(service.predict)
    assert sig.parameters["conf"].default == 0.15


def test_empty_detections_on_blank_canvas(service):
    """Verify inference on a plain uniform canvas returns 0 detections cleanly."""
    blank = Image.new("RGB", (320, 320), color=(0, 0, 0))
    res = service.predict(blank, conf=0.5)

    assert res["success"] is True
    assert res["number_of_detections"] == 0
    assert res["detections"] == []
    assert res["image_width"] == 320
    assert res["image_height"] == 320


def test_bounding_box_within_dimensions(service):
    """Verify bounding boxes (if any) are strictly within image dimensions."""
    # Create an image with high contrast shapes
    arr = np.zeros((400, 500, 3), dtype=np.uint8)
    arr[50:350, 50:450] = [200, 200, 200]
    img = Image.fromarray(arr)

    # Use a very low conf to test any candidate boxes if detected
    res = service.predict(img, conf=0.05)
    for det in res["detections"]:
        bbox = det["bounding_box"]
        assert 0 <= bbox["x1"] <= res["image_width"]
        assert 0 <= bbox["y1"] <= res["image_height"]
        assert 0 <= bbox["x2"] <= res["image_width"]
        assert 0 <= bbox["y2"] <= res["image_height"]
        assert bbox["x1"] <= bbox["x2"]
        assert bbox["y1"] <= bbox["y2"]
        assert 0 <= det["confidence"] <= 1.0
        assert det["class_id"] in CLASS_MAPPING
        assert det["class_name"] == CLASS_MAPPING[det["class_id"]]


def test_invalid_image_inputs_handled(service):
    """Verify corrupt bytes, non-existent files, and bad types raise InvalidImageError."""
    # Non-existent file
    with pytest.raises(InvalidImageError):
        service.load_image("non_existent_file_path_xyz123.jpg")

    # Corrupt / random bytes
    with pytest.raises(InvalidImageError):
        service.load_image(b"not_an_image_random_bytes_here")

    # Empty bytes
    with pytest.raises(InvalidImageError):
        service.load_image(b"")

    # Unsupported object type
    with pytest.raises(InvalidImageError):
        service.load_image(12345)  # type: ignore


def test_rgba_and_mode_conversions(service):
    """Verify RGBA, Grayscale, and Palette modes convert to RGB cleanly."""
    # RGBA image with alpha transparency
    rgba_img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    loaded_rgb = service.load_image(rgba_img)
    assert loaded_rgb.mode == "RGB"
    assert loaded_rgb.size == (200, 200)

    # Grayscale image (mode L)
    l_img = Image.new("L", (150, 150), color=100)
    loaded_l = service.load_image(l_img)
    assert loaded_l.mode == "RGB"
    assert loaded_l.size == (150, 150)


def test_annotated_image_generation_non_destructive(service):
    """Verify annotate_image does not mutate input and draws boxes cleanly."""
    original = Image.new("RGB", (400, 300), color=(240, 240, 240))
    original_copy = original.copy()

    mock_detections = [
        {
            "class_id": 0,
            "class_name": "plastic",
            "confidence": 0.88,
            "bounding_box": {"x1": 50, "y1": 50, "x2": 200, "y2": 220},
        },
        {
            "class_id": 1,
            "class_name": "metal",
            "confidence": 0.94,
            "bounding_box": {"x1": 220, "y1": 80, "x2": 380, "y2": 260},
        },
    ]

    annotated = service.annotate_image(original, mock_detections)

    # Verify return type and dimensions
    assert isinstance(annotated, Image.Image)
    assert annotated.size == original.size
    assert annotated.mode == "RGB"

    # Verify original image was NOT modified
    assert np.array_equal(np.array(original), np.array(original_copy))

    # Verify annotated image has differences (pixels were drawn)
    assert not np.array_equal(np.array(annotated), np.array(original))


def test_agnostic_nms_parameter(service):
    """Verify agnostic_nms parameter runs cleanly for both True and False."""
    dummy = Image.new("RGB", (100, 100), color=(50, 50, 50))

    res_false = service.predict(dummy, conf=0.5, agnostic_nms=False)
    assert res_false["success"] is True

    res_true = service.predict(dummy, conf=0.5, agnostic_nms=True)
    assert res_true["success"] is True
