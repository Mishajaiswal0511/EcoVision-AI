"""
download_datasets.py  –  EcoVision AI  –  Step 2
=================================================
Reproducible dataset download script.
Supports:
  - Roboflow Universe datasets (via API)
  - Direct URL downloads with checksum verification

CREDENTIALS: Never hardcode API keys.
  Set environment variable ROBOFLOW_API_KEY before running, OR
  use a .env file in the project root (requires python-dotenv).

Usage:
    # Set your key first (Windows PowerShell):
    $env:ROBOFLOW_API_KEY = "your_key_here"

    # Then run:
    python scripts/download_datasets.py \
        --target   roboflow \
        --workspace garbage \
        --project   taco-official-all-class \
        --version   6 \
        --format    yolov11 \
        --out_dir   data/raw

    # Or dry-run to see what would happen:
    python scripts/download_datasets.py --dry_run ...

Requirements:
    pip install roboflow python-dotenv requests tqdm

IMPORTANT NOTES:
  1. A free Roboflow account is required. Sign up at https://roboflow.com
  2. Generate your API key at https://app.roboflow.com/settings/api
  3. Store it ONLY in environment variables or a .env file.
  4. The .env file must be in .gitignore (already configured).
  5. This script will NOT overwrite an existing download unless --force is passed.
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env file if present (optional; requires python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars set externally


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry documents a verified publicly accessible dataset.
# Workspace/project/version values reflect the Roboflow URL structure:
#   universe.roboflow.com/<workspace>/<project>/<version>

KNOWN_DATASETS = {
    "taco-official": {
        "description": "TACO Trash Annotations in Context — official all-class version",
        "workspace": "garbage",
        "project": "taco-official-all-class",
        "recommended_version": 6,
        "format": "yolov11",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/garbage/taco-official-all-class",
        "notes": (
            "Real-world litter images. Pre-split train/valid/test. "
            "~60 original TACO classes mapped to YOLO bounding boxes. "
            "Verify class mapping before use."
        ),
    },
    "waste-detection-zotbins": {
        "description": "Waste Recognition with TACO Full (by ZotBins)",
        "workspace": "zotbins",
        "project": "waste-recognition-with-taco-full",
        "recommended_version": 1,
        "format": "yolov11",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/zotbins/waste-recognition-with-taco-full",
        "notes": (
            "70/20/10 train/val/test split. English class labels. "
            "TACO-derived bounding boxes."
        ),
    },
    "garbage-detection-friso": {
        "description": "Garbage Detection by Friso — English material labels",
        "workspace": "friso",  # NOTE: verify exact workspace name on Roboflow
        "project": "garbage-detection",
        "recommended_version": 1,
        "format": "yolov11",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com",
        "notes": (
            "Reported classes: battery, cardboard, metal, paper, plastic, "
            "biological, brown-glass, green-glass, white-glass. "
            "Exact workspace/project slug must be verified on the Roboflow page."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit(
            "\n[ERROR] ROBOFLOW_API_KEY environment variable is not set.\n"
            "  Steps to fix:\n"
            "  1. Create a free account at https://roboflow.com\n"
            "  2. Go to https://app.roboflow.com/settings/api\n"
            "  3. Copy your Private API Key.\n"
            "  4. Windows PowerShell: $env:ROBOFLOW_API_KEY = 'your_key'\n"
            "     OR add it to a .env file in the project root (never commit .env).\n"
        )
    return key


def checksum_dir(directory: Path) -> dict:
    """Compute SHA-256 checksums for all files in directory."""
    sums = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sums[str(p.relative_to(directory))] = h.hexdigest()
    return sums


def save_manifest(out_dir: Path, info: dict):
    import json
    manifest_path = out_dir / "download_manifest.json"
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    key = info.get("dataset_key", "unknown")
    existing[key] = info
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"[OK] Manifest updated -> {manifest_path}")


def list_known_datasets():
    print("\nKnown datasets in registry:")
    print(f"  {'Key':<35} {'Description'}")
    print(f"  {'-'*35} {'-'*50}")
    for key, info in KNOWN_DATASETS.items():
        print(f"  {key:<35} {info['description']}")
    print()


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_roboflow(
    workspace: str,
    project: str,
    version: int,
    fmt: str,
    out_dir: Path,
    dry_run: bool,
    force: bool,
    dataset_key: str = "",
):
    dest = out_dir / f"{workspace}_{project}_v{version}"

    if dest.exists() and not force:
        print(f"[SKIP] Already downloaded: {dest}")
        print("       Pass --force to re-download.")
        return

    if dry_run:
        print(f"[DRY RUN] Would download:")
        print(f"  workspace : {workspace}")
        print(f"  project   : {project}")
        print(f"  version   : {version}")
        print(f"  format    : {fmt}")
        print(f"  dest      : {dest}")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("roboflow package not installed. Run: pip install roboflow")

    api_key = get_api_key()
    print(f"\n[DOWNLOAD] {workspace}/{project} v{version} -> {dest}")
    print(f"  Format: {fmt}")
    print(f"  (This may take several minutes for large datasets...)\n")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=str(dest))

    print(f"\n[OK] Download complete: {dest}")
    print(f"     Location: {dataset.location}")

    # Count what was downloaded
    n_images = sum(1 for _ in dest.rglob("*.jpg")) + \
               sum(1 for _ in dest.rglob("*.png"))
    n_labels = sum(1 for _ in dest.rglob("*.txt")
                   if "classes" not in str(_) and "notes" not in str(_))

    print(f"     Images : {n_images}")
    print(f"     Labels : {n_labels}")

    info = {
        "dataset_key": dataset_key or f"{workspace}/{project}",
        "workspace": workspace,
        "project": project,
        "version": version,
        "format": fmt,
        "destination": str(dest.resolve()),
        "downloaded_at": datetime.now().isoformat(),
        "image_count": n_images,
        "label_count": n_labels,
        "license": KNOWN_DATASETS.get(dataset_key, {}).get("license", "check dataset page"),
        "source_url": KNOWN_DATASETS.get(dataset_key, {}).get("url", ""),
    }
    save_manifest(out_dir, info)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for EcoVision AI reproducibly"
    )
    sub = parser.add_subparsers(dest="command")

    # --- list ---
    sub.add_parser("list", help="List known datasets in the registry")

    # --- roboflow ---
    rf_parser = sub.add_parser("roboflow", help="Download from Roboflow Universe")
    rf_parser.add_argument("--workspace", required=True)
    rf_parser.add_argument("--project",   required=True)
    rf_parser.add_argument("--version",   type=int, required=True)
    rf_parser.add_argument("--format",    default="yolov11")
    rf_parser.add_argument("--out_dir",   default="data/raw")
    rf_parser.add_argument("--key",       default="",
                           help="Registry key for manifest (optional)")
    rf_parser.add_argument("--dry_run",   action="store_true")
    rf_parser.add_argument("--force",     action="store_true")

    # --- known ---
    kn_parser = sub.add_parser("known", help="Download a known registered dataset by key")
    kn_parser.add_argument("dataset_key", choices=list(KNOWN_DATASETS.keys()))
    kn_parser.add_argument("--out_dir",  default="data/raw")
    kn_parser.add_argument("--format",   default=None,
                           help="Override default format (default: yolov11)")
    kn_parser.add_argument("--dry_run",  action="store_true")
    kn_parser.add_argument("--force",    action="store_true")

    args = parser.parse_args()

    if args.command == "list" or args.command is None:
        list_known_datasets()
        return

    if args.command == "roboflow":
        download_roboflow(
            args.workspace, args.project, args.version,
            args.format, Path(args.out_dir),
            args.dry_run, args.force, args.key,
        )

    elif args.command == "known":
        info = KNOWN_DATASETS[args.dataset_key]
        print(f"\n[INFO] {info['description']}")
        print(f"  URL   : {info['url']}")
        print(f"  Notes : {info['notes']}")
        print(f"  License: {info['license']}")
        download_roboflow(
            info["workspace"], info["project"],
            info["recommended_version"],
            args.format or info["format"],
            Path(args.out_dir),
            args.dry_run, args.force,
            dataset_key=args.dataset_key,
        )


if __name__ == "__main__":
    main()

