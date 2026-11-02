"""
Backend services package for EcoVision AI.
"""

from backend.services.inference import (
    CLASS_COLORS,
    CLASS_MAPPING,
    InferenceError,
    InvalidImageError,
    YOLOInferenceService,
    get_inference_service,
)
from backend.services.disposal import (
    DisposalService,
    get_disposal_service,
)

__all__ = [
    "CLASS_COLORS",
    "CLASS_MAPPING",
    "InferenceError",
    "InvalidImageError",
    "YOLOInferenceService",
    "get_inference_service",
    "DisposalService",
    "get_disposal_service",
]
