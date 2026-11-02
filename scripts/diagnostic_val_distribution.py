import os
from pathlib import Path

val_lbl_dir = Path("data/processed_v3/labels/val")

single_obj = []
multi_2_3 = []
dense_4_plus = []
backgrounds = []
small_obj_images = []

taco_count = 0
gw_count = 0
other_count = 0

for p in val_lbl_dir.glob("*.txt"):
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    n = len(lines)
    stem = p.stem
    if stem.startswith("taco_"):
        taco_count += 1
    elif stem.startswith("gw_"):
        gw_count += 1
    else:
        other_count += 1

    if n == 0:
        backgrounds.append(stem)
    elif n == 1:
        single_obj.append(stem)
    elif 2 <= n <= 3:
        multi_2_3.append(stem)
    else:
        dense_4_plus.append(stem)

    has_small = False
    for l in lines:
        parts = l.split()
        if len(parts) >= 5:
            w, h = float(parts[3]), float(parts[4])
            if (w * h) < 0.005:
                has_small = True
                break
    if has_small:
        small_obj_images.append(stem)

print(f"Total validation labels: {len(list(val_lbl_dir.glob('*.txt')))}")
print(f"Source breakdown: TACO={taco_count}, GeneralWaste={gw_count}, Other={other_count}")
print(f"Single-object (n=1)       : {len(single_obj)} images ({len(single_obj)/3007*100:.1f}%)")
print(f"Multi-object (n=2..3)     : {len(multi_2_3)} images ({len(multi_2_3)/3007*100:.1f}%)")
print(f"Dense multi-object (n>=4) : {len(dense_4_plus)} images ({len(dense_4_plus)/3007*100:.1f}%)")
print(f"Background (n=0)          : {len(backgrounds)} images ({len(backgrounds)/3007*100:.1f}%)")
print(f"Images with small objects : {len(small_obj_images)} images ({len(small_obj_images)/3007*100:.1f}%)")

