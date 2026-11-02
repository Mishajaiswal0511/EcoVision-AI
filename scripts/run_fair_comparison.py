import json
import os
import time
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

CLASSES = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]

def evaluate_model(model_path: str, imgsz: int, name: str, data_yaml: str):
    print(f"\n--- Evaluating {name} (imgsz={imgsz}) ---")
    model = YOLO(model_path)
    t0 = time.time()
    val_results = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=0.15,
        device="cpu",  # Local fair CPU evaluation
        verbose=False
    )
    eval_duration = time.time() - t0

    rd = val_results.results_dict
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall = float(rd.get("metrics/recall(B)", 0.0))
    map50 = float(rd.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(rd.get("metrics/mAP50-95(B)", 0.0))

    per_class_ap50 = [float(x) for x in val_results.box.ap50] if hasattr(val_results.box, "ap50") else []
    per_class_ap = [float(x) for x in val_results.box.ap] if hasattr(val_results.box, "ap") else []

    res = {
        "name": name,
        "model_path": model_path,
        "imgsz": imgsz,
        "eval_duration_sec": round(eval_duration, 2),
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "per_class_ap50": {CLASSES[i]: per_class_ap50[i] for i in range(min(len(CLASSES), len(per_class_ap50)))},
        "per_class_map": {CLASSES[i]: per_class_ap[i] for i in range(min(len(CLASSES), len(per_class_ap)))}
    }
    return res

def run_benchmarks(model_path: str, imgsz: int, sample_images: list):
    print(f"\n--- CPU Latency & RAM Benchmark: {Path(model_path).name} (imgsz={imgsz}) ---")
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 ** 2)

    t_load_start = time.perf_counter()
    model = YOLO(model_path)
    load_time_ms = (time.perf_counter() - t_load_start) * 1000

    # Warmup
    t_warm_start = time.perf_counter()
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        _ = model.predict(dummy, imgsz=imgsz, device="cpu", verbose=False)
    warmup_time_ms = (time.perf_counter() - t_warm_start) * 1000

    # Measured runs across fixed validation sample
    latencies = []
    for img_p in sample_images:
        t0 = time.perf_counter()
        _ = model.predict(img_p, imgsz=imgsz, conf=0.15, device="cpu", verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    ram_after = process.memory_info().rss / (1024 ** 2)
    ram_delta = ram_after - ram_before

    return {
        "load_time_ms": round(load_time_ms, 2),
        "warmup_time_ms": round(warmup_time_ms, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "ram_usage_mb": round(ram_after, 2),
        "ram_delta_mb": round(ram_delta, 2)
    }

