"""
EcoVision AI — Unified Production Backend Entrypoint
Mounts the active 7-class YOLO11s + Gemini Vision AI application.
"""

import os
import sys
from pathlib import Path

# Add staging backend directory to path
staging_dir = Path(__file__).resolve().parent / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
if str(staging_dir) not in sys.path:
    sys.path.insert(0, str(staging_dir))

from main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    port_str = os.getenv("PORT", "10000")
    try:
        port = int(port_str)
    except ValueError:
        port = 10000
    print(f"Starting EcoVision AI backend on port {port}...")
    uvicorn.run("run_backend:app", host="0.0.0.0", port=port, reload=False)
