import os, random

def box_iou(b1, b2):
    # b1, b2 are [cx, cy, w, h] normalized
    x1_min, x1_max = b1[0] - b1[2]/2, b1[0] + b1[2]/2
    y1_min, y1_max = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    x2_min, x2_max = b2[0] - b2[2]/2, b2[0] + b2[2]/2
    y2_min, y2_max = b2[1] - b2[3]/2, b2[1] + b2[3]/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

total_pairs = 0
overlapping_pairs = 0
heavy_overlap_pairs = 0 # iou > 0.3
images_with_overlap = 0
total_multi_obj_images = 0

all_files = []
for split in ['train', 'val']:
    p = os.path.join('data/processed_v3/labels', split)
    for f in os.listdir(p):
        if f.endswith('.txt'):
            all_files.append(os.path.join(p, f))

random.seed(42)
sample_files = random.sample(all_files, 2500)

for fpath in sample_files:
    boxes = []
    with open(fpath, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    
    if len(boxes) > 1:
        total_multi_obj_images += 1
        has_overlap = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total_pairs += 1
                iou = box_iou(boxes[i], boxes[j])
                if iou > 0.05:
                    overlapping_pairs += 1
                    has_overlap = True
                if iou > 0.30:
                    heavy_overlap_pairs += 1
        if has_overlap:
            images_with_overlap += 1

print(f"Sampled {total_multi_obj_images} multi-object images:")
print(f"  Images with overlapping GT boxes (IoU > 0.05): {images_with_overlap} ({images_with_overlap/total_multi_obj_images*100:.1f}%)")
print(f"  Total box pairs evaluated: {total_pairs}")
print(f"  Pairs with overlap (IoU > 0.05): {overlapping_pairs} ({overlapping_pairs/total_pairs*100:.1f}%)")
print(f"  Pairs with heavy overlap (IoU > 0.30): {heavy_overlap_pairs} ({heavy_overlap_pairs/total_pairs*100:.1f}%)")

