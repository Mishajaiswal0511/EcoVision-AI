import os, random

sizes = {'small': 0, 'medium': 0, 'large': 0}
total_boxes = 0

all_files = []
for split in ['train', 'val']:
    p = os.path.join('data/processed_v3/labels', split)
    for f in os.listdir(p):
        if f.endswith('.txt'):
            all_files.append(os.path.join(p, f))

random.seed(42)
sample_files = random.sample(all_files, 2000)

for fpath in sample_files:
    with open(fpath, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                total_boxes += 1
                if area < 0.0025:
                    sizes['small'] += 1
                elif area < 0.0225:
                    sizes['medium'] += 1
                else:
                    sizes['large'] += 1

sm_p = sizes['small'] / total_boxes * 100
md_p = sizes['medium'] / total_boxes * 100
lg_p = sizes['large'] / total_boxes * 100
print(f"Sampled {total_boxes} bounding boxes:")
print(f"  Small (<32x32 px at 640): {sizes['small']} ({sm_p:.1f}%)")
print(f"  Medium (32x32 to 96x96): {sizes['medium']} ({md_p:.1f}%)")
print(f"  Large (>96x96 px at 640): {sizes['large']} ({lg_p:.1f}%)")

