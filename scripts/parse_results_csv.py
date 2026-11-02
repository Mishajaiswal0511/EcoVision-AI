import csv

def analyze():
    csv_path = r'models/experiments/yolo11m_v1/runs/train/yolo11m_waste_960/results.csv'
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            cleaned = {k.strip(): float(v.strip()) if v.strip() else 0.0 for k, v in r.items()}
            rows.append(cleaned)

    print(f"Total completed epochs recorded in results.csv: {len(rows)}")
    best_row = max(rows, key=lambda x: x.get('metrics/mAP50-95(B)', 0))
    print(f"--- BEST CHECKPOINT TELEMETRY ---")
    print(f"Best Epoch: {int(best_row['epoch'])}")
    print(f"mAP50-95  : {best_row.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"mAP50     : {best_row.get('metrics/mAP50(B)', 0):.4f}")
    print(f"Precision : {best_row.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall    : {best_row.get('metrics/recall(B)', 0):.4f}")
    print(f"Box Loss  : {best_row.get('train/box_loss', 0):.4f}")
    print(f"Cls Loss  : {best_row.get('train/cls_loss', 0):.4f}")
    print(f"DFL Loss  : {best_row.get('train/dfl_loss', 0):.4f}")

    print("\n--- LAST 5 RECORDED EPOCHS ---")
    for r in rows[-5:]:
        ep = int(r['epoch'])
        p = r.get('metrics/precision(B)', 0)
        rec = r.get('metrics/recall(B)', 0)
        m50 = r.get('metrics/mAP50(B)', 0)
        m95 = r.get('metrics/mAP50-95(B)', 0)
        print(f"Epoch {ep:2d}: Precision={p:.4f}, Recall={rec:.4f}, mAP50={m50:.4f}, mAP50-95={m95:.4f}")

if __name__ == '__main__':
    analyze()

