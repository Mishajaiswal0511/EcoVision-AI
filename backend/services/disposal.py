"""
EcoVision AI — Disposal Guidance Service
Loads and provides official waste disposal guidelines from docs/disposal_guide.json.

Classes supported:
  - plastic
  - metal
  - cardboard
  - glass
  - paper
  - general_trash
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class DisposalService:
    """Service for retrieving waste segregation and disposal guidance."""

    _instance: Optional["DisposalService"] = None

    def __new__(cls, guide_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super(DisposalService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, guide_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        if guide_path is None:
            # Default location: docs/disposal_guide.json relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            guide_path = base_dir / "docs" / "disposal_guide.json"

        self.guide_path = Path(guide_path)
        self.classes_data: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_guide()
        self._initialized = True

    def _load_guide(self) -> None:
        """Parses the JSON file, ignoring comments at the start."""
        if not self.guide_path.exists():
            raise FileNotFoundError(f"Disposal guide not found at: {self.guide_path}")

        raw_text = self.guide_path.read_text(encoding="utf-8")
        # Strip header comment lines
        lines = [line for line in raw_text.splitlines() if not line.strip().startswith("#")]
        clean_json_str = "\n".join(lines)

        try:
            parsed = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse disposal guide JSON: {e}") from e

        self.metadata = {
            "version": parsed.get("version"),
            "updated": parsed.get("updated"),
            "schema": parsed.get("schema", {}),
        }

        # Store classes in lowercase for case-insensitive lookup
        raw_classes = parsed.get("classes", {})
        self.classes_data = {
            str(k).strip().lower(): v for k, v in raw_classes.items()
        }

    def get_disposal_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves disposal guidance for a given class name.
        Matching is case-insensitive and trims whitespace.

        Returns:
            Dictionary of guidance fields, or None if category is unknown.
        """
        if not class_name:
            return None

        normalized_key = str(class_name).strip().lower()
        class_entry = self.classes_data.get(normalized_key)
        if not class_entry:
            return None

        # Return only fields supported by the JSON reference without inventing data
        return {
            "class_name": normalized_key,
            "material": class_entry.get("material"),
            "bin_color_india": class_entry.get("bin_color_india"),
            "bin_label": class_entry.get("bin_label"),
            "preparation_steps": class_entry.get("preparation_steps", []),
            "disposal_locations": class_entry.get("disposal_locations", []),
            "safety_warning": class_entry.get("safety_warning"),
            "recyclable": class_entry.get("recyclable"),
            "compositable": class_entry.get("compositable"),
            "hazardous": class_entry.get("hazardous"),
        }

    def list_supported_classes(self) -> list[str]:
        """Returns the list of supported class names."""
        return list(self.classes_data.keys())


# Singleton instance accessor
_disposal_service_instance: Optional[DisposalService] = None


def get_disposal_service(guide_path: Optional[Path] = None) -> DisposalService:
    """Returns the cached singleton DisposalService instance."""
    global _disposal_service_instance
    if _disposal_service_instance is None:
        _disposal_service_instance = DisposalService(guide_path=guide_path)
    return _disposal_service_instance

