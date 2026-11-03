"""
EcoVision AI (Staging) — FastAPI Application
REST API for AI-powered 7-class waste detection, classification, and segregation guidance.
Configured for YOLO11s V5 Candidate @ 800px on CPU.
"""

import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import sys
# Add staging backend directory to path if needed
staging_backend_dir = Path(__file__).resolve().parent
if str(staging_backend_dir) not in sys.path:
    sys.path.insert(0, str(staging_backend_dir))

from services.disposal import StagingDisposalService, get_staging_disposal_service
from services.inference import (
    CLASS_MAPPING,
    InferenceError,
    InvalidImageError,
    StagingYOLOInferenceService,
    get_staging_inference_service,
)

# Configuration
API_TITLE = "EcoVision AI API (V5 Staging)"
API_VERSION = "5.0.0-staging"
DEFAULT_ROOT = Path(__file__).resolve().parents[6]
WORKSPACE = Path(os.getenv("WORKSPACE", str(DEFAULT_ROOT)))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "training" / "yolo11s_v5_best.pt")))
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
    app.state.model_load_error = None
    app.state.disposal_load_error = None

    try:
        get_staging_disposal_service()
    except Exception as e:
        app.state.disposal_load_error = str(e)

    # Preload model in background thread so uvicorn binds to PORT in <100ms
    async def _preload_model():
        try:
            await asyncio.to_thread(get_staging_inference_service, model_path=MODEL_PATH, device="cpu")
        except Exception as e:
            app.state.model_load_error = str(e)

    asyncio.create_task(_preload_model())
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="EcoVision AI V5 7-Class Staging Service (YOLO11s @ 800px).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _image_to_base64_data_url(image: Image.Image, format: str = "JPEG") -> str:
    img = image.copy()
    if max(img.size) > 800:
        img.thumbnail((800, 800))
    buffer = io.BytesIO()
    if format.upper() == "PNG":
        img.save(buffer, format="PNG", optimize=True)
        fmt = "png"
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        fmt = "jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


@app.get("/api/health", summary="Health and Model Readiness Check")
async def health_check() -> Dict[str, Any]:
    model_load_error = getattr(app.state, "model_load_error", None)
    model_loaded = (model_load_error is None)
    disposal_load_error = getattr(app.state, "disposal_load_error", None)
    disposal_loaded = (disposal_load_error is None)

    overall_status = "healthy" if (model_loaded and disposal_loaded) else "degraded"

    return {
        "status": overall_status,
        "environment": "staging",
        "api_version": API_VERSION,
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "target_resolution": 800,
        "classes_count": len(CLASS_MAPPING),
        "classes": list(CLASS_MAPPING.values()),
        "disposal_service_loaded": disposal_loaded,
        "model_load_error": model_load_error,
        "disposal_load_error": disposal_load_error,
    }


@app.get("/api/disposal/{class_name}", summary="Get Disposal Guidance for a Category")
async def get_disposal_guidance(class_name: str) -> Dict[str, Any]:
    disposal_service = get_staging_disposal_service()
    info = disposal_service.get_disposal_info(class_name)
    if info.get("material") == "Unknown Object":
        supported = disposal_service.list_supported_classes()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disposal guidance not found for category '{class_name}'. Supported categories: {supported}",
        )
    return {
        "success": True,
        "class_name": class_name.lower(),
        "guidance": info,
    }


@app.post("/api/detect", summary="Detect Waste and Get Segregation Guidance")
async def detect_waste(
    file: UploadFile = File(..., description="Image file to analyze"),
    conf: float = Form(0.15, description="Confidence threshold (0.0 to 1.0, default 0.15)"),
    iou: float = Form(0.45, description="IoU threshold for NMS (0.0 to 1.0, default 0.45)"),
    imgsz: int = Form(800, description="Input inference image size (default 800)"),
    agnostic_nms: bool = Form(True, description="Whether to use class-agnostic NMS (default True)"),
) -> Dict[str, Any]:
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

    try:
        inference_service = get_staging_inference_service(model_path=MODEL_PATH, device="cpu")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {exc}",
        )

    try:
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

    disposal_service = get_staging_disposal_service()
    detections = prediction_result.get("detections", [])

    for det in detections:
        class_name = det.get("class_name", "")
        guidance = disposal_service.get_disposal_info(class_name)
        det["disposal"] = guidance

    try:
        annotated_pil = await asyncio.to_thread(
            inference_service.annotate_image,
            image_input=image_bytes,
            detections=detections,
        )
        annotated_data_url = _image_to_base64_data_url(annotated_pil, format="JPEG")
    except Exception:
        annotated_data_url = ""

    return {
        "success": True,
        "image_width": prediction_result["image_width"],
        "image_height": prediction_result["image_height"],
        "inference_time_ms": prediction_result["inference_time_ms"],
        "number_of_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated_data_url,
        "image_quality": prediction_result.get("image_quality"),
        "diagnostic": prediction_result.get("diagnostic"),
    }

