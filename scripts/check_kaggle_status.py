import os
import subprocess
import json
from pathlib import Path

def check_status():
    print("=" * 60)
    print("  EcoVision AI — Kaggle Environment & Auth Status Check")
    print("=" * 60)

    # 1. CLI Check
    cli_path = Path(".venv/Scripts/kaggle.exe")
    if not cli_path.exists():
        print("ERROR: Kaggle CLI not found in .venv/Scripts/kaggle.exe")
        return False

    res = subprocess.run([str(cli_path), "--version"], capture_output=True, text=True)
    print(f"Kaggle CLI Version : {res.stdout.strip()}")

    # 2. Auth test
    res_auth = subprocess.run([str(cli_path), "datasets", "list", "--mine", "--max-size", "1"], capture_output=True, text=True)
    if res_auth.returncode != 0:
        print("Authentication Status: PENDING / NOT AUTHENTICATED")
        print("Action Needed        : Please run in terminal: .\\.venv\\Scripts\\kaggle.exe auth login")
        return False
    
    print("Authentication Status: AUTHENTICATED & ACTIVE!")

    # 3. Discover Username safely
    username = None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        username = api.get_config_value('username') or api.config_values.get('username')
    except Exception:
        pass

    if not username:
        # Try finding from datasets list
        lines = [l for l in res_auth.stdout.splitlines() if "/" in l]
        if lines:
            first_ref = lines[0].split()[0]
            if "/" in first_ref:
                username = first_ref.split("/")[0]

    if username:
        print(f"Kaggle Username/Slug : {username}")
    else:
        print("Kaggle Username/Slug : (Authenticated, username can be auto-detected upon kernel creation)")

    return True

if __name__ == "__main__":
    check_status()

