# Dataset Preparation — Google Colab Instructions
# EcoVision AI · Step 2

This file explains how to run all data preparation scripts inside a
**Google Colab notebook** so the resulting processed dataset can be
used directly for GPU training in Step 3.

---

## Prerequisites

1. A free **Roboflow** account at https://roboflow.com
2. Your Roboflow **Private API Key** (Settings → API Keys)
3. A **Google Drive** account (to persist downloaded data across sessions)

---

## Step A — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directory on Drive
import os
PROJECT_DIR = '/content/drive/MyDrive/EcoVisionAI'
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"Project directory: {PROJECT_DIR}")
```

---

## Step B — Clone or Upload Project Code

Option 1: Upload the `scripts/` directory and `data/` config files directly.
Option 2: If you have a GitHub repo, clone it:

```bash
!git clone https://github.com/YOUR_USERNAME/ecovision-ai.git /content/ecovision-ai
%cd /content/ecovision-ai
```

---

## Step C — Install Dependencies

```bash
!pip install -q roboflow pillow imagehash pyyaml tqdm python-dotenv pytest
```

---

## Step D — Set API Key (Colab Secrets — NEVER hardcode)

```python
# Method 1: Colab Secrets (recommended — click the 🔑 icon in left sidebar)
from google.colab import userdata
import os
os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')

# Method 2: Manual entry (less secure — clears when runtime resets)
# import os
# os.environ['ROBOFLOW_API_KEY'] = input("Enter Roboflow API Key: ")
```

---

## Step E — Download Primary Dataset (TACO Official)

```bash
# List available registered datasets
!python scripts/download_datasets.py list

# Download the recommended dataset (dry-run first to verify)
!python scripts/download_datasets.py known taco-official --out_dir data/raw --dry_run

# If dry-run looks correct, run without --dry_run:
!python scripts/download_datasets.py known taco-official --out_dir data/raw
```

Expected output:
- `data/raw/garbage_taco-official-all-class_v6/` with `train/`, `valid/`, `test/` subdirs

---

## Step F — Audit the Raw Download

```bash
# Check what classes exist in the downloaded data
!python scripts/audit_dataset.py \
    --data_dir data/raw/garbage_taco-official-all-class_v6 \
    --classes  data/raw/garbage_taco-official-all-class_v6/data.yaml \
    --out_dir  evaluation/results \
    --preview_n 5
```

Then open `evaluation/results/audit_report.md` to check:
- Exact class names (update `data/label_mapping.json` if needed)
- Class instance counts (decide whether battery is viable)
- Any corrupt images or invalid annotations

---

## Step G — Deep Annotation Check

```bash
!python scripts/check_annotations.py \
    --data_dir data/raw/garbage_taco-official-all-class_v6 \
    --classes  data/raw/garbage_taco-official-all-class_v6/data.yaml \
    --out_dir  evaluation/results
```

Review `evaluation/results/annotation_check.md`.
The script will exit with code 1 if critical errors are found.

---

## Step H — Deduplication

```bash
# Run pHash deduplication (threshold 10 recommended)
!python scripts/deduplicate_dataset.py \
    --data_dir  data/raw/garbage_taco-official-all-class_v6 \
    --threshold 10 \
    --out_dir   evaluation/results
```

Review `evaluation/results/dedup_report.md`.
**Do not delete files automatically.** Manually remove confirmed duplicates
from the raw directory after reading the dedup plan.

---

## Step I — Label Remapping

Edit `data/label_mapping.json` to match the actual class names found in Step F.
Then run:

```bash
!python scripts/label_mapper.py \
    --in_dir    data/raw/garbage_taco-official-all-class_v6 \
    --out_dir   data/raw/taco-remapped \
    --mapping   data/label_mapping.json \
    --src_yaml  data/raw/garbage_taco-official-all-class_v6/data.yaml
```

Review `data/raw/taco-remapped/label_mapping_report.json`.
If `unmapped > 0`, update the mapping file and re-run.

---

## Step J — Create Splits

```bash
!python scripts/prepare_splits.py \
    --source     data/raw/taco-remapped \
    --out_dir    data/processed \
    --classes    data/dataset.yaml \
    --train      0.70 \
    --val        0.15 \
    --test       0.15 \
    --seed       42 \
    --dedup_plan evaluation/results/dedup_plan.json
```

After this step:
- `data/processed/images/train/`, `val/`, `test/` are populated.
- `data/processed/split_manifest.json` records every image's assignment.
- `data/processed/class_distribution.json` shows per-split class counts.
- **The test split is now SEALED. Do not use it for tuning.**

---

## Step K — Run Automated Tests

```bash
!pip install -q pytest
!pytest tests/test_data_preparation.py -v
```

All tests should pass before proceeding to Step 3 training.

---

## Step L — Copy to Google Drive for Persistence

```python
import shutil

# Processed dataset
shutil.copytree('/content/ecovision-ai/data/processed',
                f'{PROJECT_DIR}/data/processed',
                dirs_exist_ok=True)

# Evaluation results and manifests
shutil.copytree('/content/ecovision-ai/evaluation/results',
                f'{PROJECT_DIR}/evaluation/results',
                dirs_exist_ok=True)

print("Data saved to Google Drive.")
```

---

## Verification Checklist Before Step 3

- [ ] `audit_report.md` reviewed; no corrupt images
- [ ] `annotation_check.md` shows 0 critical errors
- [ ] `dedup_report.md` reviewed; cross-split contamination = 0
- [ ] `label_mapping_report.json` shows `unmapped = 0`
- [ ] `class_distribution.json` checked; all target classes are present
- [ ] `split_manifest.json` saved and committed
- [ ] Test split NOT used for any tuning
- [ ] All pytest tests pass
- [ ] `data/dataset.yaml` nc and names match final class count

---

*EcoVision AI · Step 2 Colab Guide · 2026-10-28*

