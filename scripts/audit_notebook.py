import json

def audit():
    nb_path = 'training/kaggle/ecovision_yolo11m_training.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")
    assert len(nb['cells']) == 19, f"Expected 19 cells, got {len(nb['cells'])}"

    # Cell 6 check
    cell_6 = ''.join(nb['cells'][6]['source'])
    assert 'os.symlink' in cell_6, "symlink missing in cell 6"
    assert "shutil.copytree(src_root / 'labels'" in cell_6, "labels copy missing in cell 6"
    assert "assert n_img == exp_img" in cell_6, "count assertions missing in cell 6"
    assert "Writable label directory verified" in cell_6, "cache check missing in cell 6"
    print("Cell 6 verified successfully!")

    # Cell 8 check
    cell_8 = ''.join(nb['cells'][8]['source'])
    assert "dataset_root.resolve()" in cell_8, "dataset_root resolve missing in cell 8"
    assert "'nc': 6" in cell_8, "classes missing in cell 8"
    print("Cell 8 verified successfully!")

    # Cell 12 check
    cell_12 = ''.join(nb['cells'][12]['source'])
    assert "'epochs': 100" in cell_12, "epochs missing in cell 12"
    assert "'imgsz': 960" in cell_12, "imgsz missing in cell 12"
    assert "'patience': 25" in cell_12, "patience missing in cell 12"
    assert "'mixup': 0.15" in cell_12, "mixup missing in cell 12"
    assert "'copy_paste': 0.0" in cell_12, "copy_paste missing in cell 12"
    print("Cell 12 verified successfully!")

    print("ALL NOTEBOOK AUDIT CHECKS PASSED!")

if __name__ == '__main__':
    audit()
