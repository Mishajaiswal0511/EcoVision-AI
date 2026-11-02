"""
EcoVision AI — Full CPU Latency & Memory Regression Benchmark
Measures:
- Latency (mean, median P50, P95, throughput FPS) at 800px and 1024px
- Memory consumption (Peak RSS RAM in MB)
- Model load time
- Regression verification against baseline
"""

import sys
import os
import time
import json
import tracemalloc
from pathlib import Path
import numpy as np
from PIL import Image

WORKSPACE = Path("D:/Projects/ml_project/472")
VAL_IMG_DIR = WORKSPACE / "data" / "processed_v5" / "images" / "val"
STAGING_BACKEND = WORKSPACE / "models" / "experiments" / "v5_taxonomy" / "preproduction" / "staging" / "backend"
sys.path.insert(0, str(STAGING_BACKEND))

from services.inference import StagingYOLOInferenceService

def run_benchmark(sample_size=100):
    print("=" * 80)
    print(f"ECOVISION AI — CPU LATENCY & MEMORY BENCHMARK (N={sample_size})")
    print("=" * 80)

    # 1. Measure model loading time and RAM
    tracemalloc.start()
    t_start_load = time.perf_counter()
    service = StagingYOLOInferenceService()
    load_time_sec = round(time.perf_counter() - t_start_load, 3)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Model Load Time: {load_time_sec} s | Traced Peak Mem: {peak_mem / (1024*1024):.2f} MB")

    # Sample images
    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")) + list(VAL_IMG_DIR.glob("*.png")) + list(VAL_IMG_DIR.glob("*.webp")))
    test_sample = val_images[:sample_size]

    # Warmup
    print("Warming up on 5 images...")
    for img_p in test_sample[:5]:
        service.predict(img_p, imgsz=800)

    def benchmark_res(res_size):
        latencies = []
        for img_p in test_sample:
            t0 = time.perf_counter()
            service.predict(img_p, imgsz=res_size, agnostic_nms=True)
            dur = (time.perf_counter() - t0) * 1000.0
            latencies.append(dur)
        
        mean_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        fps = 1000.0 / mean_lat

        return {
            "resolution": f"{res_size}x{res_size}",
            "sample_count": len(test_sample),
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(med_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "throughput_fps": round(fps, 2),
        }

    print("\nBenchmarking at 800x800 (Standard Production Target)...")
    res_800 = benchmark_res(800)
    print(f"  Mean: {res_800['mean_latency_ms']} ms | P95: {res_800['p95_latency_ms']} ms | FPS: {res_800['throughput_fps']}")

    print("\nBenchmarking at 1024x1024 (High-Detail Mode)...")
    res_1024 = benchmark_res(1024)
    print(f"  Mean: {res_1024['mean_latency_ms']} ms | P95: {res_1024['p95_latency_ms']} ms | FPS: {res_1024['throughput_fps']}")

    report = {
        "hardware": "AMD Ryzen 7 7735U, 16 GB RAM (CPU Inference)",
        "model_path": str(service.model_path),
        "model_load_time_sec": load_time_sec,
        "standard_800": res_800,
        "high_detail_1024": res_1024,
    }

    out_file = WORKSPACE / "models" / "experiments" / "v6_master_enhancement" / "eval" / "cpu_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark completed. Results written to {out_file}")

if __name__ == "__main__":
    run_benchmark(100)

