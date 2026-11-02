"""
EcoVision AI — Inference Service
Standalone CPU inference engine using Ultralytics YOLO11s.

Classes:
  0: plastic
  1: metal
  2: cardboard
  3: glass
  4: paper
  5: general_trash
"""

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# Diagnostic logger
logger = logging.getLogger("ecovision.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Standard class mapping
CLASS_MAPPING: Dict[int, str] = {
    0: "plastic",
    1: "metal",
    2: "cardboard",
    3: "glass",
    4: "paper",
    5: "general_trash",
}

# Visually distinct high-contrast colors per category (RGB)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (37, 99, 235),    # Blue (plastic)
    1: (217, 119, 6),    # Amber/Orange (metal)
    2: (146, 64, 14),    # Warm Brown (cardboard)
    3: (13, 148, 136),   # Teal (glass)
    4: (5, 150, 105),    # Emerald Green (paper)
    5: (225, 29, 72),    # Crimson Red (general_trash)
}

# Fallback color for unexpected classes
FALLBACK_COLOR: Tuple[int, int, int] = (100, 116, 139)  # Slate


class InferenceError(Exception):
    """Base exception for inference engine errors."""
    pass


class InvalidImageError(InferenceError):
    """Raised when an input image is corrupted, unreadable, or invalid."""
    pass


class YOLOInferenceService:
    """
    Singleton-style inference service for waste detection using YOLO11s.
    Executes on CPU for local Windows deployment.
    """

    _instance: Optional["YOLOInferenceService"] = None

    def __new__(cls, model_path: Union[str, Path] = "models/weights_best.pt", device: str = "cpu"):
        if cls._instance is None:
            cls._instance = super(YOLOInferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Union[str, Path] = "models/weights_best.pt", device: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path.resolve()}")

        # Load Ultralytics YOLO model
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise InferenceError(f"Failed to load YOLO model from {self.model_path}: {e}") from e

        # Verify class taxonomy
        self._verify_taxonomy()
        self._initialized = True

    def _verify_taxonomy(self) -> None:
        """Verify the model contains exactly the expected 6 classes in order."""
        model_names = self.model.names
        if isinstance(model_names, list):
            model_names = {i: name for i, name in enumerate(model_names)}

        if len(model_names) != len(CLASS_MAPPING):
            raise ValueError(
                f"Model taxonomy mismatch: Expected {len(CLASS_MAPPING)} classes, "
                f"got {len(model_names)} ({model_names})"
            )

        for class_id, expected_name in CLASS_MAPPING.items():
            actual_name = model_names.get(class_id)
            if actual_name != expected_name:
                raise ValueError(
                    f"Class ID {class_id} mismatch: expected '{expected_name}', got '{actual_name}'"
                )

    def load_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """
        Loads and validates an image, normalizes orientation via EXIF,
        and converts color space to standard RGB.

        Args:
            image_input: File path, raw bytes, PIL Image, or numpy array.

        Returns:
            A clean RGB PIL.Image.Image.

        Raises:
            InvalidImageError: If the image cannot be read or is invalid.
        """
        try:
            pil_image: Image.Image

            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.is_file():
                    raise InvalidImageError(f"Image file does not exist: {path}")
                try:
                    pil_image = Image.open(path)
                    pil_image.load()
                except Exception as e:
                    raise InvalidImageError(f"Failed to decode image from path '{path}': {e}") from e

            elif isinstance(image_input, (bytes, bytearray)):
                if len(image_input) == 0:
                    raise InvalidImageError("Input image bytes are empty")
                try:
                    stream = io.BytesIO(image_input)
                    pil_image = Image.open(stream)
                    pil_image.load()
                except Exception as e:
                    raise InvalidImageError(f"Failed to decode image from bytes: {e}") from e

            elif isinstance(image_input, Image.Image):
                # Work on a copy
                pil_image = image_input.copy()

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    raise InvalidImageError("Input numpy array is empty")
                try:
                    pil_image = Image.fromarray(image_input)
                except Exception as e:
                    raise InvalidImageError(f"Failed to convert numpy array to image: {e}") from e

            else:
                raise InvalidImageError(
                    f"Unsupported image input type: {type(image_input)}. "
                    "Expected str, Path, bytes, PIL.Image, or np.ndarray."
                )

            # Apply EXIF rotation if present (e.g. mobile camera photos)
            try:
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                # If EXIF reading fails or is malformed, proceed with un-rotated image
                pass

            # Convert non-RGB (RGBA, P, LA, L, CMYK) to RGB
            if pil_image.mode != "RGB":
                if pil_image.mode == "RGBA":
                    # Create white background for alpha compositing
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                else:
                    pil_image = pil_image.convert("RGB")

            # Check that image has non-zero dimensions
            if pil_image.width <= 0 or pil_image.height <= 0:
                raise InvalidImageError(
                    f"Invalid image dimensions: {pil_image.width}x{pil_image.height}"
                )

            return pil_image

        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Unexpected error loading image: {e}") from e

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        conf: float = 0.15,
        iou: float = 0.45,
        imgsz: int = 640,
        agnostic_nms: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs object detection inference on an input image.

        Args:
            image_input: File path, bytes, PIL Image, or numpy array.
            conf: Minimum confidence threshold (default 0.15).
            iou: IoU threshold for NMS (default 0.45).
            imgsz: Input inference resolution (default 640).
            agnostic_nms: Whether to use class-agnostic NMS (default False).

        Returns:
            Structured dictionary matching the EcoVision detection schema.
        """
        image = self.load_image(image_input)
        width, height = image.size

        start_time = time.perf_counter()

        try:
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                agnostic_nms=agnostic_nms,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(f"YOLO inference failed: {e}") from e

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        raw_boxes = results[0].boxes if len(results) > 0 and results[0].boxes is not None else []
        num_raw = len(raw_boxes)

        detections: List[Dict[str, Any]] = []
        if num_raw > 0:
            for box in raw_boxes:
                # Bounding box in pixel coordinates (xyxy)
                xyxy = box.xyxy[0].cpu().numpy()
                x1 = max(0.0, min(float(xyxy[0]), float(width)))
                y1 = max(0.0, min(float(xyxy[1]), float(height)))
                x2 = max(0.0, min(float(xyxy[2]), float(width)))
                y2 = max(0.0, min(float(xyxy[3]), float(height)))

                confidence = round(float(box.conf[0].cpu().numpy()), 4)
                class_id = int(box.cls[0].cpu().numpy())
                class_name = CLASS_MAPPING.get(class_id, self.model.names.get(class_id, f"class_{class_id}"))

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                })

        # Diagnostic logging
        conf_scores = [d["confidence"] for d in detections]
        class_names = [d["class_name"] for d in detections]
        logger.info(
            f"[DIAGNOSTIC] Image: {width}x{height} | conf threshold: {conf} | "
            f"Raw YOLO detections: {num_raw} | Post-filter detections: {len(detections)} | "
            f"Classes: {class_names} | Scores: {conf_scores}"
        )

        if num_raw == 0:
            # Check if any candidate boxes existed below requested threshold
            try:
                diag_low = self.model.predict(
                    source=image,
                    conf=0.01,
                    iou=iou,
                    imgsz=imgsz,
                    device=self.device,
                    verbose=False,
                )
                low_boxes = diag_low[0].boxes if len(diag_low) > 0 and diag_low[0].boxes is not None else []
                if len(low_boxes) > 0:
                    top_cand_conf = round(float(low_boxes.conf.max().cpu().numpy()), 4)
                    cand_classes = [CLASS_MAPPING.get(int(b.cls[0].cpu().numpy()), "unknown") for b in low_boxes[:5]]
                    logger.info(
                        f"[DIAGNOSTIC - ZERO DETECTIONS AT {conf}] Found {len(low_boxes)} candidate boxes below threshold. "
                        f"Highest candidate conf was {top_cand_conf} (candidates: {cand_classes}). "
                        f"Reason: All model predictions were filtered because {top_cand_conf} < {conf}."
                    )
                else:
                    logger.info(
                        f"[DIAGNOSTIC - ZERO DETECTIONS AT {conf}] 0 detections even at conf=0.01. "
                        f"Reason: Model feature activations did not trigger any of the 6 waste classes."
                    )
            except Exception:
                pass

        return {
            "success": True,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": inference_time_ms,
            "number_of_detections": len(detections),
            "detections": detections,
        }

    def annotate_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        detections: List[Dict[str, Any]],
        box_thickness: int = 3,
        font_size: int = 14,
    ) -> Image.Image:
        """
        Draws high-contrast bounding boxes and label badges onto a copy of the image.
        Does not mutate the original input image.

        Args:
            image_input: Original image.
            detections: List of detection dictionaries from `predict(...)`.
            box_thickness: Border line thickness in pixels.
            font_size: Font size for label badges.

        Returns:
            A new PIL Image with visual annotations drawn.
        """
        # Obtain clean base image without modifying original
        base_image = self.load_image(image_input)
        annotated = base_image.copy()
        draw = ImageDraw.Draw(annotated)

        # Attempt to load a clean truetype font, fallback to default
        try:
            # Common Windows fonts
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        width, height = annotated.size

        for det in detections:
            bbox = det.get("bounding_box", {})
            x1 = max(0, min(int(round(bbox.get("x1", 0))), width))
            y1 = max(0, min(int(round(bbox.get("y1", 0))), height))
            x2 = max(0, min(int(round(bbox.get("x2", 0))), width))
            y2 = max(0, min(int(round(bbox.get("y2", 0))), height))

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = det.get("class_id", -1)
            class_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            conf_pct = f"{int(round(conf * 100))}%"
            label_text = f"{class_name} {conf_pct}"

            color = CLASS_COLORS.get(class_id, FALLBACK_COLOR)

            # Draw outer bounding box
            for i in range(box_thickness):
                draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)

            # Compute label pill dimensions
            if font and hasattr(draw, "textbbox"):
                tbbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = tbbox[2] - tbbox[0]
                text_h = tbbox[3] - tbbox[1]
            elif font and hasattr(font, "getsize"):
                text_w, text_h = font.getsize(label_text)
            else:
                text_w, text_h = (len(label_text) * 7, 12)

            padding_x = 6
            padding_y = 3
            badge_w = text_w + padding_x * 2
            badge_h = text_h + padding_y * 2

            # Place badge above bounding box if space allows, otherwise inside top-left
            if y1 - badge_h >= 0:
                badge_x1 = x1
                badge_y1 = y1 - badge_h
            else:
                badge_x1 = x1
                badge_y1 = y1

            badge_x2 = min(badge_x1 + badge_w, width)
            badge_y2 = badge_y1 + badge_h

            # Draw badge background pill with category color
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)

            # Text color: high-contrast white text
            text_x = badge_x1 + padding_x
            text_y = badge_y1 + padding_y

            # If font is supported, draw text
            if font:
                draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
            else:
                draw.text((text_x, text_y), label_text, fill=(255, 255, 255))

        return annotated


# Module-level helper for singleton access
_global_service: Optional[YOLOInferenceService] = None


def get_inference_service(
    model_path: Union[str, Path] = "models/weights_best.pt",
    device: str = "cpu"
) -> YOLOInferenceService:
    """Returns the cached singleton YOLOInferenceService instance."""
    global _global_service
    if _global_service is None:
        _global_service = YOLOInferenceService(model_path=model_path, device=device)
    return _global_service

