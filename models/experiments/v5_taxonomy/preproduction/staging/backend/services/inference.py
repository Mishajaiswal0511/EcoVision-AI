"""
EcoVision AI (Staging) — Inference Service
Standalone CPU inference engine using Ultralytics YOLO11s V5 7-Class Model.

Classes:
  0: plastic
  1: metal
  2: cardboard
  3: glass
  4: paper
  5: general_trash
  6: organic
"""

import io
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

logger = logging.getLogger("ecovision.staging.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 7-Class standard taxonomy
CLASS_MAPPING: Dict[int, str] = {
    0: "plastic",
    1: "metal",
    2: "cardboard",
    3: "glass",
    4: "paper",
    5: "general_trash",
    6: "organic",
}

# Visually distinct high-contrast colors per category (RGB)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (37, 99, 235),    # Blue (plastic)
    1: (217, 119, 6),    # Amber/Orange (metal)
    2: (146, 64, 14),    # Warm Brown (cardboard)
    3: (13, 148, 136),   # Teal (glass)
    4: (5, 150, 105),    # Emerald Green (paper)
    5: (225, 29, 72),    # Crimson Red (general_trash)
    6: (22, 163, 74),    # Vibrant Green (organic)
}

FALLBACK_COLOR: Tuple[int, int, int] = (100, 116, 139)  # Slate


class InferenceError(Exception):
    pass


class InvalidImageError(InferenceError):
    pass


class StagingYOLOInferenceService:
    """
    Singleton-style inference service for waste detection using YOLO11s V5 7-class model.
    Runs on CPU.
    """

    _instance: Optional["StagingYOLOInferenceService"] = None

    def __new__(cls, model_path: Union[str, Path] = "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt", device: str = "cpu"):
        if cls._instance is None:
            cls._instance = super(StagingYOLOInferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Union[str, Path] = "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt", device: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Staging model checkpoint not found at: {self.model_path.resolve()}")

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise InferenceError(f"Failed to load staging YOLO model from {self.model_path}: {e}") from e

        self._verify_taxonomy()
        self._initialized = True

    def _verify_taxonomy(self) -> None:
        """Verify the model contains exactly the expected 7 classes in order."""
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

            try:
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                pass

            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            return pil_image

        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Unexpected error loading image: {e}") from e

    def _apply_containment_suppression(self, detections: List[Dict[str, Any]], containment_thresh: float = 0.80) -> List[Dict[str, Any]]:
        """
        Suppresses redundant aggregate bounding boxes and cross-class ghost proposals
        using geometric containment.
        """
        if len(detections) <= 1:
            return detections

        def b_area(bbox):
            return max(0.0, bbox["x2"] - bbox["x1"]) * max(0.0, bbox["y2"] - bbox["y1"])

        def b_inter(b1, b2):
            x1 = max(b1["x1"], b2["x1"])
            y1 = max(b1["y1"], b2["y1"])
            x2 = min(b1["x2"], b2["x2"])
            y2 = min(b1["y2"], b2["y2"])
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = [True] * len(sorted_dets)

        # 1. Suppress small cross-class ghost proposals inside high-confidence dominant containers
        for i in range(len(sorted_dets)):
            if not keep[i]:
                continue
            d_dom = sorted_dets[i]
            if d_dom["confidence"] >= 0.70:
                a_dom = b_area(d_dom["bounding_box"])
                for j in range(len(sorted_dets)):
                    if i == j or not keep[j]:
                        continue
                    d_cand = sorted_dets[j]
                    if d_cand["confidence"] < 0.40 and d_cand["class_name"] != d_dom["class_name"]:
                        a_cand = b_area(d_cand["bounding_box"])
                        inter = b_inter(d_dom["bounding_box"], d_cand["bounding_box"])
                        if a_cand > 0 and (inter / a_cand) >= containment_thresh and (a_cand / a_dom) < 0.20:
                            keep[j] = False

        # 2. Suppress redundant aggregate envelope boxes of same class when multiple sub-boxes exist
        for i in range(len(sorted_dets)):
            if not keep[i]:
                continue
            di = sorted_dets[i]
            ai = b_area(di["bounding_box"])
            contained_sub_boxes = 0
            for j in range(len(sorted_dets)):
                if i == j or not keep[j]:
                    continue
                dj = sorted_dets[j]
                if di["class_name"] == dj["class_name"]:
                    aj = b_area(dj["bounding_box"])
                    if aj < ai:
                        inter = b_inter(di["bounding_box"], dj["bounding_box"])
                        if aj > 0 and (inter / aj) >= containment_thresh:
                            contained_sub_boxes += 1

            if contained_sub_boxes >= 2 and di["confidence"] < 0.45:
                keep[i] = False

        return [sorted_dets[i] for i in range(len(sorted_dets)) if keep[i]]

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        conf: float = 0.15,
        iou: float = 0.45,
        imgsz: int = 800,
        agnostic_nms: bool = True,
        containment_suppression: bool = True,
    ) -> Dict[str, Any]:
        image = self.load_image(image_input)
        width, height = image.size

        start_time = time.perf_counter()

        # Image Quality Assessment (Sharpness & Lighting)
        quality_info = self._assess_image_quality(image)

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

        detections_before_containment = len(detections)
        if containment_suppression and len(detections) > 1:
            detections = self._apply_containment_suppression(detections)

        # Hybrid Vision Fallback & Arbiter: If 0 detections or ambiguous/uncertain detection (conf < 0.72),
        # consult Multimodal Vision Intelligence to accurately recognize unseen real-world waste and eliminate false classifications
        max_conf = max([d["confidence"] for d in detections], default=0.0)
        needs_vision_fallback = (
            len(detections) == 0 or
            max_conf < 0.72
        )

        if needs_vision_fallback:
            try:
                from services.vision_ai import get_vision_ai_service
                vision_service = get_vision_ai_service()
                if vision_service.is_available():
                    vision_result = vision_service.analyze_waste_image(image)
                    if vision_result:
                        logger.info(f"Vision AI recovered waste detection: {vision_result['class_name']} ({vision_result['confidence']})")
                        detections = [vision_result]
            except Exception as e:
                logger.warning(f"Vision fallback failed: {e}")

        # Smart Diagnostic Analysis for No-Detection (or low detections)
        diagnostic = self._generate_scan_diagnostic(
            image=image,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=imgsz,
            final_detections=detections,
            detections_before_containment=detections_before_containment,
            quality_info=quality_info,
        )

        return {
            "success": True,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": inference_time_ms,
            "number_of_detections": len(detections),
            "detections": detections,
            "image_quality": quality_info,
            "diagnostic": diagnostic,
        }

    def _assess_image_quality(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Calculates objective, observable image quality metrics:
        - Brightness (mean grayscale pixel intensity)
        - Sharpness (Laplacian variance of grayscale image)
        """
        try:
            import cv2
            np_img = np.array(pil_image)
            if len(np_img.shape) == 3:
                gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = np_img

            brightness = float(np.mean(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            return {
                "brightness": round(brightness, 1),
                "is_underexposed": brightness < 45.0,
                "is_overexposed": brightness > 230.0,
                "sharpness_score": round(laplacian_var, 1),
                "is_blurry": laplacian_var < 50.0,
            }
        except Exception as e:
            logger.warning(f"Image quality assessment failed: {e}")
            return {
                "brightness": 128.0,
                "is_underexposed": False,
                "is_overexposed": False,
                "sharpness_score": 100.0,
                "is_blurry": False,
            }

    def _generate_scan_diagnostic(
        self,
        image: Image.Image,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        final_detections: List[Dict[str, Any]],
        detections_before_containment: int,
        quality_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generates evidence-based Smart Assistant explanations:
        - When detections exist: provides summary and material insight.
        - When 0 detections: checks low-conf proposals (conf=0.03), image quality, and NMS suppression.
        """
        num_final = len(final_detections)

        if num_final > 0:
            # Successful detection explanation
            class_counts = {}
            for d in final_detections:
                c = d["class_name"]
                class_counts[c] = class_counts.get(c, 0) + 1

            top_class = max(class_counts, key=class_counts.get)
            top_conf = max(d["confidence"] for d in final_detections if d["class_name"] == top_class)

            summary_text = (
                f"Successfully localized {num_final} waste object{'s' if num_final > 1 else ''}. "
                f"Dominant material detected is {top_class.replace('_', ' ').title()} "
                f"with peak model confidence of {top_conf * 100:.1f}%."
            )

            return {
                "status": "success",
                "primary_reason": "DETECTION_CONFIRMED",
                "title": "Waste Items Successfully Localized",
                "explanation": summary_text,
                "suggestions": [
                    "Review the category-specific disposal instructions in the table below.",
                    "Ensure dry recyclables are rinsed and segregated from organic/wet waste.",
                ],
            }

        # ZERO FINAL DETECTIONS CASE
        # Check if containment suppression removed the last box
        if detections_before_containment > 0 and num_final == 0:
            return {
                "status": "no_detection",
                "primary_reason": "CONTAINMENT_SUPPRESSION",
                "title": "Redundant Bounding Box Suppressed",
                "explanation": (
                    "The model found a candidate proposal, but CA-NMS geometric suppression "
                    "filtered it out because it overlapped completely inside another conflicting region."
                ),
                "suggestions": [
                    "Try isolating the single item on a clear, clean background.",
                    "Reposition the camera to show the object separated from other containers.",
                ],
            }

        # Priority 1: Check for severe physical image degradation (Lighting & Blur)
        # Blur or dark lighting degrades feature extraction at the root before trusting sub-threshold predictions
        if quality_info.get("is_underexposed"):
            return {
                "status": "no_detection",
                "primary_reason": "POOR_LIGHTING_DARK",
                "title": "Insufficient Scene Lighting",
                "explanation": (
                    f"Image analysis indicates severe underexposure (average scene brightness: "
                    f"{quality_info['brightness']}/255). Low lighting obscures surface textures and logos needed for waste recognition."
                ),
                "evidence": {
                    "measured_brightness": quality_info["brightness"],
                    "lighting_status": "Severely underexposed (< 45/255)",
                },
                "suggestions": [
                    "Turn on room lighting or use your camera flash.",
                    "Move the waste object into direct, even lighting.",
                    "Avoid heavy backlighting and deep shadows.",
                ],
            }

        if quality_info.get("is_overexposed"):
            return {
                "status": "no_detection",
                "primary_reason": "POOR_LIGHTING_OVEREXPOSED",
                "title": "Image Overexposed or Washed Out",
                "explanation": (
                    f"Image analysis detected excessive glare and highlight blowout (brightness: "
                    f"{quality_info['brightness']}/255), washing out object edges and material color."
                ),
                "evidence": {
                    "measured_brightness": quality_info["brightness"],
                    "lighting_status": "Severely overexposed (> 230/255)",
                },
                "suggestions": [
                    "Reduce strong flash glare or reflections off shiny metal/plastic surfaces.",
                    "Adjust camera exposure to preserve material textures.",
                ],
            }

        if quality_info.get("is_blurry"):
            return {
                "status": "no_detection",
                "primary_reason": "MOTION_BLUR",
                "title": "Image Motion Blur Detected",
                "explanation": (
                    f"The image sharpness score ({quality_info['sharpness_score']}) is below the "
                    "clarity threshold, indicating camera shake, motion blur, or out-of-focus optics."
                ),
                "evidence": {
                    "sharpness_score": quality_info["sharpness_score"],
                    "clarity_threshold": 50.0,
                    "blur_status": "Severe blur detected",
                },
                "suggestions": [
                    "Hold the camera steady and wait for focus lock before snapping.",
                    "Clean your camera lens of smudges or oil.",
                    "Avoid moving the waste item while capturing.",
                ],
            }

        # Priority 2: Probe for sub-threshold candidate proposals (conf=0.03)
        sub_candidates = []
        try:
            probe_results = self.model.predict(
                source=image,
                conf=0.03,
                iou=iou_threshold,
                imgsz=imgsz,
                agnostic_nms=True,
                device=self.device,
                verbose=False,
            )
            if len(probe_results) > 0 and probe_results[0].boxes is not None:
                for b in probe_results[0].boxes:
                    sub_candidates.append({
                        "class_name": CLASS_MAPPING.get(int(b.cls[0]), "unknown"),
                        "confidence": round(float(b.conf[0]), 3),
                    })
        except Exception as e:
            logger.warning(f"Low-confidence probe failed: {e}")

        candidates_below_thresh = [c for c in sub_candidates if c["confidence"] < conf_threshold]
        if candidates_below_thresh:
            top_candidate = max(candidates_below_thresh, key=lambda x: x["confidence"])
            cand_name = top_candidate["class_name"].replace("_", " ").title()
            cand_conf = top_candidate["confidence"] * 100.0

            return {
                "status": "no_detection",
                "primary_reason": "BELOW_CONFIDENCE_THRESHOLD",
                "title": "Candidate Found Below Confidence Threshold",
                "explanation": (
                    f"The model detected features resembling {cand_name} (confidence: {cand_conf:.1f}%), "
                    f"but it was rejected by your current threshold of {conf_threshold * 100:.0f}%. "
                    "This usually happens when objects are viewed from far away or blend into the background."
                ),
                "evidence": {
                    "top_candidate_class": top_candidate["class_name"],
                    "top_candidate_confidence": top_candidate["confidence"],
                    "current_threshold": conf_threshold,
                },
                "suggestions": [
                    f"Lower the Confidence Slider in the settings panel to {max(5, int(cand_conf - 2))}% to capture this object.",
                    "Move the camera closer so the item occupies at least 50% of the image frame.",
                    "Enable High-Detail Resolution (1024px) for small or distant objects.",
                ],
            }

        # Situation 3: No candidates found at all (Unsupported object or pure background)
        return {
            "status": "no_detection",
            "primary_reason": "NO_SUPPORTED_WASTE_FOUND",
            "title": "No Supported Waste Features Identified",
            "explanation": (
                "The YOLO model analyzed the entire scene but found no features matching any of its "
                "7 supported categories (Plastic, Metal, Cardboard, Glass, Paper, General Trash, Organic). "
                "The object may be outside the supported taxonomy (e.g. textiles, electronics, hazardous chemicals) "
                "or the image contains only background."
            ),
            "suggestions": [
                "Verify whether the object belongs to one of the 7 supported classes.",
                "Frame the waste item in the center against a neutral, uncluttered background.",
                "Capture from an angle where key identifiers (openings, caps, labels, corrugated edges) are visible.",
            ],
        }

    def annotate_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        detections: List[Dict[str, Any]],
        box_thickness: int = 3,
        font_size: int = 14,
    ) -> Image.Image:
        base_image = self.load_image(image_input)
        annotated = base_image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        width, height = annotated.size

        occupied_badges: List[Tuple[int, int, int, int]] = []

        for det in detections:
            if not isinstance(det, dict):
                continue

            bbox = det.get("bounding_box")
            if not bbox or not isinstance(bbox, dict):
                continue

            x1_raw = bbox.get("x1")
            y1_raw = bbox.get("y1")
            x2_raw = bbox.get("x2")
            y2_raw = bbox.get("y2")

            coords = (x1_raw, y1_raw, x2_raw, y2_raw)
            if any(
                c is None
                or isinstance(c, bool)
                or not isinstance(c, (int, float))
                or not math.isfinite(c)
                for c in coords
            ):
                continue

            x1 = max(0, min(int(round(x1_raw)), width))
            y1 = max(0, min(int(round(y1_raw)), height))
            x2 = max(0, min(int(round(x2_raw)), width))
            y2 = max(0, min(int(round(y2_raw)), height))

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = det.get("class_id", -1)
            class_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            conf_pct = f"{int(round(conf * 100))}%"
            label_text = f"{class_name} {conf_pct}"

            color = CLASS_COLORS.get(class_id, FALLBACK_COLOR)

            for i in range(box_thickness):
                draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)

            if font and hasattr(draw, "textbbox"):
                tbbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = tbbox[2] - tbbox[0]
                text_h = tbbox[3] - tbbox[1]
            else:
                text_w, text_h = (len(label_text) * 7, 12)

            padding_x = 6
            padding_y = 3
            badge_w = text_w + padding_x * 2
            badge_h = text_h + padding_y * 2

            # Intelligent horizontal clamping to prevent labels from extending outside the canvas
            badge_x1 = max(0, min(x1, width - badge_w))
            badge_x2 = min(badge_x1 + badge_w, width)

            # Candidate vertical positions to avoid label collisions:
            # Candidate 1: Above top edge
            # Candidate 2: Below bottom edge
            # Candidate 3: Inside top edge
            cand_positions = []
            if y1 - badge_h >= 0:
                cand_positions.append(y1 - badge_h)
            if y2 + badge_h <= height:
                cand_positions.append(y2)
            cand_positions.append(min(y1, height - badge_h))

            # Helper to check collision with already drawn badges
            def has_collision(by1, by2):
                for ob in occupied_badges:
                    # check 2D overlap with a small margin
                    if not (badge_x2 <= ob[0] or badge_x1 >= ob[2] or by2 <= ob[1] or by1 >= ob[3]):
                        return True
                return False

            chosen_y1 = None
            for cand_y in cand_positions:
                cand_y2 = cand_y + badge_h
                if not has_collision(cand_y, cand_y2):
                    chosen_y1 = cand_y
                    break

            if chosen_y1 is None:
                # If all candidates collide, offset downward from the last colliding badge
                chosen_y1 = min(max(0, y1 - badge_h), height - badge_h)
                for ob in occupied_badges:
                    if not (badge_x2 <= ob[0] or badge_x1 >= ob[2]):
                        if abs(chosen_y1 - ob[1]) < badge_h:
                            chosen_y1 = min(ob[3] + 2, height - badge_h)

            badge_y1 = chosen_y1
            badge_y2 = badge_y1 + badge_h
            occupied_badges.append((badge_x1, badge_y1, badge_x2, badge_y2))

            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)

            text_x = badge_x1 + padding_x
            text_y = badge_y1 + padding_y

            if font:
                draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
            else:
                draw.text((text_x, text_y), label_text, fill=(255, 255, 255))

        return annotated


_global_staging_service: Optional[StagingYOLOInferenceService] = None


def get_staging_inference_service(
    model_path: Union[str, Path] = "models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt",
    device: str = "cpu"
) -> StagingYOLOInferenceService:
    global _global_staging_service
    if _global_staging_service is None:
        _global_staging_service = StagingYOLOInferenceService(model_path=model_path, device=device)
    return _global_staging_service

