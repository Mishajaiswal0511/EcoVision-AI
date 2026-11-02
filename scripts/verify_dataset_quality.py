import json
from pathlib import Path
from collections import defaultdict
import yaml

def verify_all():
    print("=" * 60)
    print("  ECOVISION AI - PRE-TRAINING QUALITY REVIEW AUDIT")
    print("=" * 60)

    # 1. Split integrity
    with open('evaluation/results/dedup_plan.json') as f:
        plan = json.load(f)
    with open('data/processed/split_manifest.json') as f:
        manifest = json.load(f)

    entry_splits = {e['image']: e['split'] for e in manifest['entries']}
    
    cross_groups = [g for g in plan if g.get('cross_split_contamination')]
    print(f"\n[1] Verifying 5 original cross-split leakage groups:")
    for i, g in enumerate(cross_groups, 1):
        keeper = Path(g['keeper']).name
        candidates = [Path(c['path']).name for c in g['candidates_for_removal']]
        all_members = [keeper] + candidates
        splits = {m: entry_splits.get(m) for m in all_members}
        unique_splits = set(splits.values())
        print(f"  Group {i} (original: {g['splits_present']}) -> final split: {unique_splits} (Leakage-free: {len(unique_splits) == 1})")
        assert len(unique_splits) == 1, f"Leakage detected in group {i}!"

    # Check all 120 clusters
    cluster_split_counts = defaultdict(int)
    for g in plan:
        keeper = Path(g['keeper']).name
        candidates = [Path(c['path']).name for c in g['candidates_for_removal']]
        all_members = [keeper] + candidates
        splits = {entry_splits.get(m) for m in all_members if m in entry_splits}
        assert len(splits) <= 1, f"Cluster split across boundaries: {splits}"
        for s in splits:
            cluster_split_counts[s] += 1
    print(f"  Cluster distribution across splits: {dict(cluster_split_counts)}")

    # 2. Filesystem check
    print(f"\n[2] Filesystem image and label counts verification:")
    for split in ['train', 'val', 'test']:
        img_p = Path('data/processed/images') / split
        lbl_p = Path('data/processed/labels') / split
        imgs = list(img_p.glob('*.*'))
        lbls = list(lbl_p.glob('*.txt'))
        print(f"  Split '{split}': {len(imgs)} images, {len(lbls)} labels")
        assert len(imgs) == len(lbls), f"Mismatch in {split}!"

    # 3. Annotation reconciliation
    print(f"\n[3] Annotation count reconciliation:")
    with open('data/raw/remapped_taco/remapping_log.json') as f:
        remap_log = json.load(f)

    total_src = remap_log['total_source_annotations']
    mapped = remap_log['annotations_mapped']
    excluded = remap_log['annotations_excluded']
    degen = len(remap_log['degenerate_boxes_removed'])
    print(f"  Source annotations : {total_src}")
    print(f"  Mapped annotations : {mapped}")
    print(f"  Excluded (null)    : {excluded}")
    print(f"  Degenerate dropped : {degen}")
    print(f"  Sum check: {mapped} + {excluded} + {degen} = {mapped + excluded + degen} (Expected: {total_src})")
    assert mapped + excluded + degen == total_src, "Reconciliation sum mismatch!"

    # 4. Class IDs and YAML validation
    print(f"\n[4] Class IDs and YAML validation:")
    with open('data/dataset.yaml') as f:
        cfg = yaml.safe_load(f)
    yaml_names = cfg['names']
    print(f"  YAML nc: {cfg['nc']}, names count: {len(yaml_names)}")
    assert cfg['nc'] == len(yaml_names) == 6, "YAML nc != 6!"

    # Scan processed labels for class IDs
    found_ids = set()
    total_boxes = 0
    per_class_counts = defaultdict(int)
    for split in ['train', 'val', 'test']:
        lbl_p = Path('data/processed/labels') / split
        for f in lbl_p.glob('*.txt'):
            with open(f) as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        found_ids.add(cls_id)
                        total_boxes += 1
                        per_class_counts[cls_id] += 1

    print(f"  Found class IDs in labels: {sorted(found_ids)}")
    assert sorted(found_ids) == [0, 1, 2, 3, 4, 5], f"Non-contiguous class IDs: {found_ids}"
    print(f"  Total bounding boxes in processed dataset: {total_boxes}")
    assert total_boxes == mapped, f"Processed boxes {total_boxes} != mapped {mapped}"
    for cid, name in yaml_names.items():
        print(f"    Class {cid} ({name:<14}): {per_class_counts[cid]} boxes")

    print("\n[OK] All mathematical and structural checks PASSED perfectly.")

if __name__ == '__main__':
    verify_all()
