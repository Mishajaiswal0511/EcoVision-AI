"""
EcoVision AI - V5 Kaggle Training Monitor & Artifact Harvester
Monitors mishajaiswal0511/ecovision-v5-7-class-model-training headlessly.
Pulls trained checkpoints and evaluation metrics to models/experiments/v5_taxonomy/training/.
"""

import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KERNEL_ID = "mishajaiswal0511/ecovision-v5-7-class-model-training"
WORKSPACE = Path("D:/Projects/ml_project/472")
OUT_DIR = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = WORKSPACE / "training" / "kaggle_v5" / "training_execution.log"


def main():
    api = KaggleApi()
    api.authenticate()

    print("=" * 80)
    print(f"MONITORING KAGGLE TRAINING: {KERNEL_ID}")
    print(f"Destination: {OUT_DIR}")
    print("=" * 80)

    t0 = time.time()
    last_status = None

    while True:
        try:
            status_obj = api.kernels_status(KERNEL_ID)
            status = str(status_obj.status) if hasattr(status_obj, "status") else str(status_obj)
            elapsed_m = (time.time() - t0) / 60

            if status != last_status:
                print(f"[{elapsed_m:.1f}m] Kernel Status: {status}")
                last_status = status

            # Check completion
            if "COMPLETE" in status.upper():
                print(f"\n[OK] Training completed in {elapsed_m:.1f} minutes! Downloading outputs...")
                api.kernels_output(KERNEL_ID, path=str(OUT_DIR))
                print(f"Outputs successfully downloaded to {OUT_DIR}")
                break
            elif "ERROR" in status.upper() or "CANCEL" in status.upper():
                print(f"\n[FAIL] Kernel stopped with status: {status}")
                # Pull whatever logs or outputs exist
                try:
                    api.kernels_output(KERNEL_ID, path=str(OUT_DIR))
                except Exception:
                    pass
                break

            time.sleep(30)
        except Exception as e:
            print(f"Warning during polling: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()

