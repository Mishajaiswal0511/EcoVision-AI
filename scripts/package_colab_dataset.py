"""
EcoVision AI — Dataset Packager for Google Colab
================================================
Packages the processed 6-class dataset (images + labels) into a zip
archive ready for Google Drive / Google Colab upload.
"""

import os
import zipfile
from pathlib import Path

def package_dataset(processed_dir: Path, output_zip: Path):
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found at: {processed_dir}")
    
    print(f"Compressing {processed_dir} into {output_zip} ...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include data/dataset.yaml
        dataset_yaml = processed_dir.parent / "dataset.yaml"
        if dataset_yaml.exists():
            zf.write(dataset_yaml, arcname="data/dataset.yaml")
            file_count += 1
            
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                file_path = Path(root) / file
                # Store relative to workspace root: e.g. data/processed/images/train/img.jpg
                rel_path = file_path.relative_to(processed_dir.parent.parent)
                zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                file_count += 1
                if file_count % 500 == 0:
                    print(f"  Archived {file_count} files...")
                    
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Successfully created {output_zip} ({file_count} files, {zip_size_mb:.2f} MB)")

if __name__ == "__main__":
    WORKSPACE = Path(__file__).resolve().parent.parent
    PROCESSED = WORKSPACE / "data" / "processed"
    OUT_ZIP = WORKSPACE / "data" / "ecovision_processed_dataset.zip"
    package_dataset(PROCESSED, OUT_ZIP)

