"""
EcoVision AI (Staging) — Disposal Guidance Service
Loads and provides official waste disposal guidelines from staging data/disposal_guide.json.
7 Classes supported:
  - plastic
  - metal
  - cardboard
  - glass
  - paper
  - general_trash
  - organic
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class StagingDisposalService:
    """Service for retrieving waste segregation and disposal guidance in staging."""

    _instance: Optional["StagingDisposalService"] = None

    def __new__(cls, guide_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super(StagingDisposalService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, guide_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        if guide_path is None:
            # Default location in staging
            base_dir = Path(__file__).resolve().parent.parent
            guide_path = base_dir / "data" / "disposal_guide.json"

        self.guide_path = Path(guide_path)
        self.classes_data: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_guide()
        self._initialized = True

    def _load_guide(self) -> None:
        """Parses the JSON file, ignoring comments at the start."""
        if not self.guide_path.exists():
            raise FileNotFoundError(f"Staging disposal guide not found at: {self.guide_path}")

        raw_text = self.guide_path.read_text(encoding="utf-8")
        lines = [line for line in raw_text.splitlines() if not line.strip().startswith("#")]
        clean_json_str = "\n".join(lines)

        try:
            parsed = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse disposal guide JSON: {e}") from e

        self.metadata = {
            "version": parsed.get("version"),
            "updated": parsed.get("updated"),
            "governing_policy": parsed.get("governing_policy"),
        }

        raw_classes = parsed.get("classes", {})
        self.classes_data = {
            str(k).strip().lower(): v for k, v in raw_classes.items()
        }

    def get_disposal_info(self, class_name: str) -> Dict[str, Any]:
        """Retrieves disposal details for a waste category."""
        normalized = str(class_name).strip().lower()
        if normalized in self.classes_data:
            return self.classes_data[normalized]

        return {
            "material": "Unknown Object",
            "bin_color_india": "Grey / Black",
            "bin_label": "Unclassified Waste",
            "preparation_steps": ["Inspect item material manually before disposal"],
            "disposal_locations": ["Municipal drop-off"],
            "safety_warning": "⚠️ Item not matched in automated guide. Segregate with caution.",
            "recyclable": False,
            "compositable": False,
            "hazardous": False,
        }

    def list_supported_classes(self) -> list:
        return list(self.classes_data.keys())


def get_staging_disposal_service(guide_path: Optional[Path] = None) -> StagingDisposalService:
    return StagingDisposalService(guide_path=guide_path)

