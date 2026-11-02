import json
from pathlib import Path

nb_path = Path('training/kaggle/ecovision_yolo11m_training.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_6_code = """import os
import time
import zipfile
import shutil
from pathlib import Path

t_start = time.time()
input_dir = Path('/kaggle/input')
print(f"Inspecting input directory: {input_dir}")
for p in input_dir.iterdir():
    print(f"  Available input source: {p.name}")

# Check for uncompressed processed_v3 first (standard Kaggle unpack)
processed_dirs = list(input_dir.glob('**/processed_v3'))
dataset_root = None

if processed_dirs:
    src_root = processed_dirs[0]
    print(f"Found uncompressed dataset at: {src_root}")
    target_dir = Path('/tmp/dataset/data/processed_v3')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Symlink images directory (read-only access is sufficient, instant 0-second operation)
    images_dst = target_dir / 'images'
    if not images_dst.exists():
        print(f"Symlinking {src_root / 'images'} -> {images_dst} ...")
        os.symlink(src_root / 'images', images_dst)
    
    # 2. Copy labels directory (~6 MB total text files, provides local writable directory for YOLO train.cache)
    labels_dst = target_dir / 'labels'
    if not labels_dst.exists():
        print(f"Copying labels {src_root / 'labels'} -> {labels_dst} ...")
        shutil.copytree(src_root / 'labels', labels_dst)
        
    dataset_root = target_dir
else:
    zip_candidates = list(input_dir.glob('**/*.zip'))
    if zip_candidates:
        zip_path = zip_candidates[0]
        print(f"Found dataset archive: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
        extract_dir = Path('/tmp/dataset')
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        dataset_root = extract_dir / 'data' / 'processed_v3' if (extract_dir / 'data' / 'processed_v3').exists() else extract_dir
    else:
        raise FileNotFoundError(f"Could not find processed_v3 or zip in {input_dir}!")

t_elapsed = time.time() - t_start
print(f"Dataset staging successfully completed in {t_elapsed:.2f} seconds!")
print(f"Verified dataset root: {dataset_root}")

# Verify resolution and file counts
EXPECTED_COUNTS = {'train': (8242, 8242), 'val': (3007, 3007)}
for split in ['train', 'val']:
    img_files = list((dataset_root / 'images' / split).glob('*.*'))
    lbl_files = list((dataset_root / 'labels' / split).glob('*.txt'))
    n_img = len(img_files)
    n_lbl = len(lbl_files)
    print(f"Split [{split:5s}]: {n_img:5d} images | {n_lbl:5d} labels")
    exp_img, exp_lbl = EXPECTED_COUNTS[split]
    assert n_img == exp_img, f"Count mismatch in {split} images: got {n_img}, expected {exp_img}"
    assert n_lbl == exp_lbl, f"Count mismatch in {split} labels: got {n_lbl}, expected {exp_lbl}"

# Test write permissions on labels directory for .cache creation
test_cache = dataset_root / 'labels' / '.cache_write_test'
test_cache.write_text('cache_ok')
test_cache.unlink()
print("Writable label directory verified: Ultralytics can generate .cache files without permission errors.")

# Security assertions: test and real_world_eval MUST be strictly absent
test_dir = dataset_root / 'images' / 'test'
assert not test_dir.exists() or len(list(test_dir.glob('*.*'))) == 0, "SECURITY ERROR: Test data detected in training environment!"
rwe_dirs = list(input_dir.glob('**/*real_world_eval*')) + list(dataset_root.glob('**/*real_world_eval*'))
assert len(rwe_dirs) == 0, "SECURITY ERROR: Real-world eval detected in environment!"
print("Security check passed: Test split and real-world eval are strictly absent.")
"""

nb['cells'][6]['source'] = [line + '\n' for line in new_cell_6_code.split('\n')]
if nb['cells'][6]['source'] and nb['cells'][6]['source'][-1].endswith('\n'):
    nb['cells'][6]['source'][-1] = nb['cells'][6]['source'][-1][:-1]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated Cell 6 in", nb_path)

