"""
EcoVision AI — Unified Backend Entrypoint
Mounts the active 7-class YOLO11s + Gemini Vision AI application.
"""

import sys
from pathlib import Path

# Add staging backend directory to path
staging_dir = Path(__file__).resolve().parent / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
if str(staging_dir) not in sys.path:
    sys.path.insert(0, str(staging_dir))

from main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_backend:app", host="0.0.0.0", port=8000, reload=False)
