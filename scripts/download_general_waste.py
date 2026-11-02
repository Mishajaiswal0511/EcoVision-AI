"""
download_general_waste.py  –  EcoVision AI
=========================================
Downloads mcl2/general-waste-data version 1 from Roboflow Universe
in YOLOv11 format directly to data/raw/general_waste_data.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

WORKSPACE = Path(__file__).resolve().parent.parent
ENV_PATH = WORKSPACE / ".env"
TARGET_DIR = WORKSPACE / "data" / "raw" / "general_waste_data"

if not ENV_PATH.exists():
    sys.exit(f"ERROR: .env file not found at {ENV_PATH}")

load_dotenv(ENV_PATH)
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    sys.exit("ERROR: ROBOFLOW_API_KEY not found in .env")

print(f"============================================================")
print(f"  Downloading General Waste Data (mcl2/general-waste-data/1)")
print(f"  Destination : {TARGET_DIR}")
print(f"============================================================\n")

try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("mcl2").project("general-waste-data")
    version = project.version(1)
    
    # Must pass overwrite=True because TARGET_DIR exists
    dataset = version.download("yolov11", location=str(TARGET_DIR), overwrite=True)
    print(f"\n[OK] Download complete to {dataset.location}")
except Exception as e:
    sys.exit(f"ERROR downloading dataset: {e}")
