"""
EcoVision AI — FastAPI Application
REST API for AI-powered waste detection, classification, and segregation guidance.
"""

import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from backend.services.disposal import DisposalService, get_disposal_service
from backend.services.inference import (
    CLASS_MAPPING,
    InferenceError,
    InvalidImageError,
    YOLOInferenceService,
    get_inference_service,
)

# Configuration
API_TITLE = "EcoVision AI API"
API_VERSION = "5.0.0-user-testing"
MODEL_PATH = os.getenv("MODEL_PATH", "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt")
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "15"))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize model and disposal service at startup."""
    # Pre-warm services
    try:
        get_inference_service(model_path=MODEL_PATH, device="cpu")
    except Exception as e:
        # We record error but don't crash startup so health endpoint can report status
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None

    try:
        get_disposal_service()
    except Exception as e:
        app.state.disposal_load_error = str(e)
    else:
        app.state.disposal_load_error = None

    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Intelligent waste detection and segregation guidance powered by YOLO11s.",
    lifespan=lifespan,
)

# CORS Middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _image_to_base64_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Encodes a PIL Image to a base64 Data URL string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


@app.get("/api/health", summary="Health and Model Readiness Check")
async def health_check() -> Dict[str, Any]:
    """
    Returns the operational status of the API, application version,
    and YOLO model readiness.
    """
    model_service: Optional[YOLOInferenceService] = None
    model_loaded = False
    model_classes: Dict[int, str] = {}
    model_architecture = "YOLO11s"
    inference_device = "cpu"
    error_message: Optional[str] = getattr(app.state, "model_load_error", None)

    try:
        model_service = get_inference_service(model_path=MODEL_PATH, device="cpu")
        model_loaded = model_service is not None and model_service.model is not None
        if model_loaded:
            model_classes = {int(k): v for k, v in model_service.model.names.items()}
            inference_device = str(model_service.device)
    except Exception as exc:
        model_loaded = False
        error_message = str(exc)

    is_healthy = model_loaded and error_message is None

    return {
        "status": "healthy" if is_healthy else "degraded",
        "api_status": "online",
        "model_loaded": model_loaded,
        "model_architecture": model_architecture,
        "model_version": "V5-Candidate (7-Class)" if len(model_classes) == 7 else "V4-Production (6-Class)",
        "active_checkpoint": str(MODEL_PATH),
        "target_resolution": DEFAULT_IMGSZ,
        "device": inference_device,
        "num_classes": len(model_classes) if model_loaded else len(CLASS_MAPPING),
        "class_names": model_classes if model_loaded else CLASS_MAPPING,
        "version": API_VERSION,
        "error": error_message,
    }


@app.get("/api/disposal/{class_name}", summary="Get Disposal Guidance for a Category")
async def get_disposal_guidance(class_name: str) -> Dict[str, Any]:
    """
    Retrieves official disposal and segregation guidance for a specific waste class.
    Class matching is case-insensitive.
    """
    disposal_service = get_disposal_service()
    info = disposal_service.get_disposal_info(class_name)

    if info is None:
        supported = disposal_service.list_supported_classes()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disposal guidance not found for category '{class_name}'. Supported categories: {supported}",
        )

    return info


@app.post("/api/detect", summary="Detect Waste and Get Segregation Guidance")
async def detect_waste(
    file: UploadFile = File(..., description="Image file to analyze"),
    conf: float = Form(0.15, description="Confidence threshold (0.0 to 1.0, default 0.15)"),
    iou: float = Form(0.45, description="IoU threshold for NMS (0.0 to 1.0, default 0.45)"),
    imgsz: int = Form(DEFAULT_IMGSZ, description="Input inference image size (positive integer, default 800)"),
    agnostic_nms: bool = Form(False, description="Whether to use class-agnostic NMS"),
) -> Dict[str, Any]:
    """
    Analyzes an uploaded image, detects waste items, generates an annotated preview,
    and attaches category-specific disposal instructions.
    """
    # 1. Parameter Validation
    if not (0.0 < conf <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid confidence threshold: conf must be between 0.0 and 1.0, got {conf}",
        )

    if not (0.0 <= iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid IoU threshold: iou must be between 0.0 and 1.0, got {iou}",
        )

    if imgsz <= 0 or imgsz > 2048:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image size: imgsz must be between 32 and 2048, got {imgsz}",
        )

    # 2. File size & Content-Type sanity check
    # Check filename extension if provided
    filename = (file.filename or "").lower()
    has_valid_extension = any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS)

    content_type = (file.content_type or "").lower()
    has_valid_mime = content_type in SUPPORTED_MIME_TYPES

    if filename and not (has_valid_extension or has_valid_mime):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format: '{file.filename}' with Content-Type '{file.content_type}'. "
                f"Supported extensions: {sorted(list(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # 3. Read image bytes into memory without permanent disk storage
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image size ({len(image_bytes)} bytes) exceeds the maximum allowed limit of {MAX_IMAGE_SIZE_MB} MB.",
        )

    # 4. Execute CPU inference in threadpool to keep asyncio event loop responsive
    try:
        inference_service = get_inference_service(model_path=MODEL_PATH, device="cpu")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {exc}",
        )

    try:
        # Run inference in a background thread
        prediction_result = await asyncio.to_thread(
            inference_service.predict,
            image_input=image_bytes,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            agnostic_nms=agnostic_nms,
        )
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unreadable image: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc}",
        )

    # 5. Attach disposal guidance to each detected item
    disposal_service = get_disposal_service()
    detections = prediction_result.get("detections", [])

    for det in detections:
        class_name = det.get("class_name", "")
        guidance = disposal_service.get_disposal_info(class_name)
        det["disposal"] = guidance

    # 6. Generate non-destructive annotated preview image in threadpool
    try:
        annotated_pil = await asyncio.to_thread(
            inference_service.annotate_image,
            image_input=image_bytes,
            detections=detections,
        )
        annotated_data_url = _image_to_base64_data_url(annotated_pil, format="PNG")
    except Exception as exc:
        # If visual annotation fails, return raw empty string without breaking the response
        annotated_data_url = ""

    return {
        "success": True,
        "image_width": prediction_result["image_width"],
        "image_height": prediction_result["image_height"],
        "inference_time_ms": prediction_result["inference_time_ms"],
        "number_of_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated_data_url,
    }
