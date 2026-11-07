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
Detect up to 8 prominent distinct waste objects visible in the image.
For each individual object, provide a tight bounding box and assign it to exactly one of these 7 taxonomy classes:
[plastic, metal, cardboard, glass, paper, general_trash, organic].

Guidelines:
1. "organic": Food scraps, fruit/vegetable peels, eggshells, leftovers, leaves.
2. "plastic": Bottles, cups, tubs, caps, clean containers, film wrappers.
3. "metal": Beverage cans, food tins, clean foil, metal caps.
4. "cardboard": Corrugated shipping boxes, paperboard cartons, cup sleeves.
5. "glass": Beverage bottles, jars, intact container glass.
6. "paper": Disposable paper coffee cups, office paper, newspapers, paper bags.
7. "general_trash": Greasy pizza boxes, used napkins/tissues, chip packets, styrofoam.

Return ONLY a valid JSON list of objects:
[
  {
    "class_name": "plastic",
    "confidence": 0.95,
    "item_name": "clear plastic cup",
    "box_2d": [ymin, xmin, ymax, xmax]
  }
]
Where box_2d is [ymin, xmin, ymax, xmax] as 0 to 1000 integer coordinates tightly bounding that individual object.
Never return a single box around the entire scene if individual items can be localized.
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
                self.client = genai.Client(api_key=self.api_key, http_options={"timeout": 25000})
                logger.info("Google Gemini Vision AI client initialized with 25s timeout.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None

        self._initialized = True

    def is_available(self) -> bool:
        return self.client is not None

    def _parse_item(self, item: Dict[str, Any], width: int, height: int) -> Optional[Dict[str, Any]]:
        try:
            class_name = item.get("class_name", "").lower().strip()
            if class_name not in VALID_CLASSES:
                for valid in VALID_CLASSES:
                    if valid in class_name:
                        class_name = valid
                        break
                else:
                    class_name = "general_trash"

            class_id = VALID_CLASSES[class_name]
            confidence = float(item.get("confidence", 0.95))
            item_name = item.get("item_name", item.get("detected_items", [class_name]))
            box_2d = item.get("box_2d", [100, 100, 900, 900])

            ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
            max_coord = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))

            if max_coord <= 1.05 and max_coord > 0:
                x1 = round(xmin * width, 1)
                y1 = round(ymin * height, 1)
                x2 = round(xmax * width, 1)
                y2 = round(ymax * height, 1)
            elif max_coord <= 1005:
                x1 = round((xmin / 1000.0) * width, 1)
                y1 = round((ymin / 1000.0) * height, 1)
                x2 = round((xmax / 1000.0) * width, 1)
                y2 = round((ymax / 1000.0) * height, 1)
            else:
                x1, y1, x2, y2 = round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                return None

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
                "item_name": item_name,
                "source": "Gemini-Vision-Intelligence",
            }
        except Exception as e:
            logger.warning(f"Failed to parse item: {e}")
            return None

    def analyze_waste_image(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Analyzes image using Gemini 2.5 Flash and returns a list of localized waste items.
        """
        if not self.is_available():
            return []

        try:
            img = pil_image.copy()
            if max(img.size) > 640:
                img.thumbnail((640, 640))

            models_to_try = [
                "gemini-3.5-flash-lite",
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-2.5-flash",
            ]
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
                return []

            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            width, height = pil_image.size
            results = []

            if isinstance(data, list):
                for item in data:
                    parsed = self._parse_item(item, width, height)
                    if parsed:
                        results.append(parsed)
            elif isinstance(data, dict):
                items = data.get("detections", [data])
                for item in items:
                    parsed = self._parse_item(item, width, height)
                    if parsed:
                        results.append(parsed)

            return results
        except Exception as e:
            logger.error(f"Gemini Vision analysis error: {e}")
            return []


def get_vision_ai_service() -> VisionAIService:
    return VisionAIService()

