"""
audit_multiobject_dataset.py — Comprehensive Dataset Audit for Multi-Object Waste Detection
"""
import glob
import os
import json
from collections import Counter, defaultdict

data_dir = 'data/processed_v3'
splits = ['train', 'val', 'test']
class_names = {0: 'plastic', 1: 'metal', 2: 'cardboard', 3: 'glass', 4: 'paper', 5: 'general_trash'}

total_objects = 0
objects_per_split = defaultdict(int)
class_counts = defaultdict(lambda: defaultdict(int))
images_per_class = defaultdict(lambda: defaultdict(int))
obj_count_dist = defaultdict(lambda: Counter())
class_cooccurrence = defaultdict(lambda: Counter())
source_breakdown = defaultdict(lambda: Counter())
source_obj_dist = defaultdict(lambda: Counter())
source_class_dist = defaultdict(lambda: Counter())
bbox_sizes = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})

for s in splits:
    lbl_dir = os.path.join(data_dir, 'labels', s)
    lbl_files = glob.glob(os.path.join(lbl_dir, '*.txt'))
    
    for lp in lbl_files:
        fname = os.path.basename(lp)
        source = 'neg' if fname.startswith('neg_') else ('gw' if fname.startswith('gw_') else 'taco')
        source_breakdown[s][source] += 1
        
        with open(lp, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            
        n_objs = len(lines)
        obj_count_dist[s][n_objs] += 1
        source_obj_dist[source][n_objs] += 1
        objects_per_split[s] += n_objs
        total_objects += n_objs
        
        classes_in_img = set()
        for line in lines:
            parts = line.split()
            c_id = int(parts[0])
            w = float(parts[3])
            h = float(parts[4])
            area = w * h # normalized area: small < 0.0025 (32x32 on 640x640), medium < 0.0225 (96x96), large >= 0.0225
            if area < 0.0025:
                bbox_sizes[s]['small'] += 1
            elif area < 0.0225:
                bbox_sizes[s]['medium'] += 1
            else:
                bbox_sizes[s]['large'] += 1
                
            class_counts[s][c_id] += 1
            source_class_dist[source][c_id] += 1
            classes_in_img.add(c_id)
            
        for c_id in classes_in_img:
            images_per_class[s][c_id] += 1
            
        n_classes = len(classes_in_img)
        class_cooccurrence[s][n_classes] += 1

print('=== TOTAL OBJECTS BY SPLIT ===')
for s in splits:
    n_imgs = sum(obj_count_dist[s].values())
    n_objs = objects_per_split[s]
    print(f"{s}: {n_objs} boxes in {n_imgs} images (mean {n_objs/n_imgs:.2f} boxes/img)")
print(f"TOTAL: {total_objects} boxes in {sum(objects_per_split.values())} total")

print('\n=== SOURCE BREAKDOWN (IMAGES) ===')
for s in splits:
    print(f"{s}: {dict(source_breakdown[s])}")

print('\n=== OBJECTS PER SOURCE ===')
for src in ['taco', 'gw', 'neg']:
    tot_src_imgs = sum(source_obj_dist[src].values())
    tot_src_boxes = sum(k * v for k, v in source_obj_dist[src].items())
    print(f"Source '{src}': {tot_src_boxes} boxes in {tot_src_imgs} images (mean {tot_src_boxes/tot_src_imgs if tot_src_imgs else 0:.2f} boxes/img)")
    print(f"  Distribution: 0={source_obj_dist[src][0]}, 1={source_obj_dist[src][1]}, 2={source_obj_dist[src][2]}, 3-5={sum(source_obj_dist[src][i] for i in range(3,6))}, 6-10={sum(source_obj_dist[src][i] for i in range(6,11))}, >10={sum(source_obj_dist[src][i] for i in range(11, 200))}")

print('\n=== CLASS INSTANCE COUNTS (BOXES) ===')
for c_id in range(6):
    name = class_names[c_id]
    c_train = class_counts['train'][c_id]
    c_val = class_counts['val'][c_id]
    c_test = class_counts['test'][c_id]
    c_tot = c_train + c_val + c_test
    print(f"Class {c_id} ({name:<13}): train={c_train:<6} val={c_val:<6} test={c_test:<6} total={c_tot:<6} ({c_tot/total_objects*100:.1f}%)")

print('\n=== IMAGES PER CLASS ===')
total_images = sum(sum(source_breakdown[s].values()) for s in splits)
for c_id in range(6):
    name = class_names[c_id]
    im_train = images_per_class['train'][c_id]
    im_val = images_per_class['val'][c_id]
    im_test = images_per_class['test'][c_id]
    im_tot = im_train + im_val + im_test
    print(f"Class {c_id} ({name:<13}): train={im_train:<6} val={im_val:<6} test={im_test:<6} total={im_tot:<6} ({im_tot/total_images*100:.1f}% of images)")

print('\n=== BBOX SIZES (COCO normalized area thresholds) ===')
for s in splits:
    tot_s = sum(bbox_sizes[s].values())
    sm = bbox_sizes[s]['small']
    md = bbox_sizes[s]['medium']
    lg = bbox_sizes[s]['large']
    print(f"{s}: small={sm} ({sm/tot_s*100:.1f}%), medium={md} ({md/tot_s*100:.1f}%), large={lg} ({lg/tot_s*100:.1f}%)")
tot_sm = sum(bbox_sizes[s]['small'] for s in splits)
tot_md = sum(bbox_sizes[s]['medium'] for s in splits)
tot_lg = sum(bbox_sizes[s]['large'] for s in splits)
print(f"TOTAL: small={tot_sm} ({tot_sm/total_objects*100:.1f}%), medium={tot_md} ({tot_md/total_objects*100:.1f}%), large={tot_lg} ({tot_lg/total_objects*100:.1f}%)")

print('\n=== OBJECT COUNT PER IMAGE DISTRIBUTION (ACROSS SPLITS) ===')
for s in splits:
    c_dist = obj_count_dist[s]
    n_img = sum(c_dist.values())
    print(f"Split {s} ({n_img} images):")
    print(f"  0 objects (neg): {c_dist[0]} ({c_dist[0]/n_img*100:.1f}%)")
    print(f"  1 object:        {c_dist[1]} ({c_dist[1]/n_img*100:.1f}%)")
    print(f"  2 objects:       {c_dist[2]} ({c_dist[2]/n_img*100:.1f}%)")
    print(f"  3-5 objects:     {sum(c_dist[i] for i in range(3,6))} ({sum(c_dist[i] for i in range(3,6))/n_img*100:.1f}%)")
    print(f"  6-10 objects:    {sum(c_dist[i] for i in range(6,11))} ({sum(c_dist[i] for i in range(6,11))/n_img*100:.1f}%)")
    print(f"  >10 objects:     {sum(c_dist[i] for i in range(11,200))} ({sum(c_dist[i] for i in range(11,200))/n_img*100:.1f}%)")

print('\n=== UNIQUE CLASSES PER IMAGE DISTRIBUTION ===')
for s in splits:
    co = class_cooccurrence[s]
    n_img = sum(co.values())
    print(f"Split {s}: 0 classes={co[0]} ({co[0]/n_img*100:.1f}%), 1 class={co[1]} ({co[1]/n_img*100:.1f}%), 2 classes={co[2]} ({co[2]/n_img*100:.1f}%), >=3 classes={sum(co[i] for i in range(3,10))} ({sum(co[i] for i in range(3,10))/n_img*100:.1f}%)")

tot_co = Counter()
for s in splits:
    tot_co += class_cooccurrence[s]
print(f"OVERALL ALL DATA: 0 classes={tot_co[0]} ({tot_co[0]/total_images*100:.1f}%), 1 class={tot_co[1]} ({tot_co[1]/total_images*100:.1f}%), 2 classes={tot_co[2]} ({tot_co[2]/total_images*100:.1f}%), >=3 classes={sum(tot_co[i] for i in range(3,10))} ({sum(tot_co[i] for i in range(3,10))/total_images*100:.1f}%)")

