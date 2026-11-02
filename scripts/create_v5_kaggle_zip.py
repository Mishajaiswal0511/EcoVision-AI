"""
EcoVision AI - V5 Kaggle Dataset Zipper
Creates data/kaggle_upload_v5/ecovision_v5_dataset.zip
"""

import sys
import zipfile
from pathlib import Path
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path("D:/Projects/ml_project/472")
V5_DIR = WORKSPACE / "data" / "processed_v5"
UPLOAD_DIR = WORKSPACE / "data" / "kaggle_upload_v5"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = UPLOAD_DIR / "ecovision_v5_dataset.zip"


def main():
    print(f"Creating V5 Kaggle dataset zip at {ZIP_PATH}...")
    t0 = time.time()

    meta_json = {
        "title": "EcoVision V5 7-Class Waste Dataset",
        "id": "mishajaiswal0511/ecovision-v5-dataset",
        "licenses": [{"name": "CC-BY-4.0"}]
    }
    (UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")

    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # Add dataset_v5.yaml at root
        zf.write(V5_DIR / "dataset.yaml", arcname="data/processed_v5/dataset.yaml")
        zf.write(V5_DIR / "dataset_manifest.json", arcname="data/processed_v5/dataset_manifest.json")
        count += 2

        for split in ["train", "val"]:
            img_dir = V5_DIR / "images" / split
            lbl_dir = V5_DIR / "labels" / split

            for img_p in img_dir.glob("*.*"):
                arcname = f"data/processed_v5/images/{split}/{img_p.name}"
                zf.write(img_p, arcname=arcname)
                count += 1
                if count % 2000 == 0:
                    print(f"  Zipped {count} files ({time.time() - t0:.1f}s)...")

            for lbl_p in lbl_dir.glob("*.txt"):
                arcname = f"data/processed_v5/labels/{split}/{lbl_p.name}"
                zf.write(lbl_p, arcname=arcname)
                count += 1
                if count % 2000 == 0:
                    print(f"  Zipped {count} files ({time.time() - t0:.1f}s)...")

    duration = time.time() - t0
    size_mb = ZIP_PATH.stat().st_size / (1024 ** 2)
    print(f"\nDone! Zipped {count} files in {duration:.1f}s. Zip size: {size_mb:.2f} MB")


if __name__ == "__main__":
    import json
    main()

