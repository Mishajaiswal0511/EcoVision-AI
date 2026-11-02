"""
EcoVision AI — Multimodal Vision Intelligence Service
Powered by Google Gemini 2.5 Flash for high-accuracy waste classification,
OOD recovery, and verification of complex mixed/unseen waste items.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ecovision.vision_ai")

# Valid 7-class taxonomy
VALID_CLASSES = {
    "plastic": 0,
    "metal": 1,
    "cardboard": 2,
    "glass": 3,
    "paper": 4,
    "general_trash": 5,
    "organic": 6,
}

PROMPT_TEMPLATE = """
You are the AI Vision Core of the EcoVision AI waste segregation system.
Analyze the provided waste image.
Classify the waste into exactly one of these 7 taxonomy classes:
[plastic, metal, cardboard, glass, paper, general_trash, organic].

Guidelines:
1. "organic": Raw food trimmings, fruit/vegetable peels, eggshells, coffee grounds, cooked food leftovers, garden leaves, bread, bones.
2. "plastic": Bottles, tubs, caps, clean containers, films.
3. "metal": Beverage cans, food tins, clean aluminium foil, metal caps.
4. "cardboard": Corrugated shipping boxes, clean paperboard cartons.
5. "glass": Beverage bottles, jars, intact container glass.
6. "paper": Office paper, newspapers, magazines, paper bags.
7. "general_trash": Greasy pizza boxes, used napkins/tissues, chip packets (multilayer plastics), styrofoam, non-recyclable composites.

Return ONLY a valid JSON object matching this exact schema:
{
  "class_name": "organic",
  "confidence": 0.98,
  "detected_items": ["eggshells", "citrus peel"],
  "explanation": "Food preparation scraps and eggshells are compostable wet organic waste.",
  "box_2d": [ymin, xmin, ymax, xmax]
}
Where box_2d is [ymin, xmin, ymax, xmax] as 0 to 1000 integer coordinates tightly bounding the waste object (e.g. [150, 100, 850, 600]).
"""


class VisionAIService:
    _instance: Optional["VisionAIService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionAIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Google Gemini Vision AI client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None

        self._initialized = True

    def is_available(self) -> bool:
        return self.client is not None

    def analyze_waste_image(self, pil_image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Analyzes image using Gemini 2.5 Flash and returns structured waste classification.
        """
        if not self.is_available():
            return None

        try:
            # Resize image if excessively large to keep request fast (< 1024px)
            img = pil_image.copy()
            if max(img.size) > 1200:
                img.thumbnail((1200, 1200))

            # Try primary fast high-quota model with fallback
            models_to_try = ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-2.5-flash"]
            response = None
            last_err = None

            for m in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=[img, PROMPT_TEMPLATE],
                    )
                    if response and response.text:
                        break
                except Exception as ex:
                    last_err = ex
                    continue

            if response is None or not response.text:
                logger.error(f"All vision AI models failed: {last_err}")
                return None

            raw_text = response.text.strip()
            # Clean possible markdown block wrappers
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())

            class_name = data.get("class_name", "").lower().strip()
            if class_name not in VALID_CLASSES:
                # Fallback to closest match
                for valid in VALID_CLASSES:
                    if valid in class_name:
                        class_name = valid
                        break
                else:
                    class_name = "general_trash"

            class_id = VALID_CLASSES[class_name]
            confidence = float(data.get("confidence", 0.95))
            detected_items = data.get("detected_items", [])
            explanation = data.get("explanation", "")
            box_2d = data.get("box_2d", [100, 100, 900, 900])

            # Convert bounding box coordinates to actual image pixel space
            width, height = pil_image.size
            try:
                ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
                max_coord = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))

                if max_coord <= 1.05 and max_coord > 0:
                    # Normalized 0.0 - 1.0 coordinates
                    x1 = round(xmin * width, 1)
                    y1 = round(ymin * height, 1)
                    x2 = round(xmax * width, 1)
                    y2 = round(ymax * height, 1)
                elif max_coord <= 1005:
                    # 0 - 1000 integer coordinates
                    x1 = round((xmin / 1000.0) * width, 1)
                    y1 = round((ymin / 1000.0) * height, 1)
                    x2 = round((xmax / 1000.0) * width, 1)
                    y2 = round((ymax / 1000.0) * height, 1)
                else:
                    # Direct pixel coordinates
                    x1, y1, x2, y2 = round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)

                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # Clamp strictly within image bounds
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

                # If box is too tight or degenerated, provide a tight center-object envelope
                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    x1 = round(width * 0.15, 1)
                    y1 = round(height * 0.15, 1)
                    x2 = round(width * 0.85, 1)
                    y2 = round(height * 0.85, 1)
            except Exception as box_err:
                logger.warning(f"Error parsing box_2d {box_2d}: {box_err}")
                x1 = round(width * 0.15, 1)
                y1 = round(height * 0.15, 1)
                x2 = round(width * 0.85, 1)
                y2 = round(height * 0.85, 1)

            return {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bounding_box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "detected_items": detected_items,
                "explanation": explanation,
                "source": "Gemini-Vision-Intelligence",
            }
        except Exception as e:
            logger.error(f"Gemini Vision analysis error: {e}")
            return None


def get_vision_ai_service() -> VisionAIService:
    return VisionAIService()

