# EcoVision AI — V5 Winning Checkpoint CPU Latency Benchmark Report

**Benchmark Execution Date:** 2026-11-01 10:57:59  
**Target Checkpoint:** [`models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`](file:///D:/Projects/ml_project/472/models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt)  
**Hardware Platform:** AMD Ryzen 7 7735U (8 Physical Cores, 16 Logical Threads), 16 GB RAM, CPU Inference  
**Production Inference Config:** Device: `CPU`, Confidence: `0.15`, IoU: `0.45`  
**Sample Set:** 100 representative validation images from `data/processed_v5/images/val`  

---

## 1. Executive Summary & Verdict

| Configuration | Mean Total Latency | Median (P50) | P95 Latency | Throughput (FPS) | Peak RAM (RSS) | Assessment Verdict |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **640x640** | **137.02 ms** | **138.39 ms** | **158.73 ms** | **7.3 FPS** | **518.0 MB** | **PASS** |
| **800x800** | **171.93 ms** | **170.77 ms** | **186.22 ms** | **5.82 FPS** | **593.96 MB** | **PASS** |

> [!NOTE]
> **Real-Time Responsiveness Analysis:**
> - At **640x640**, the model delivers **7.3 FPS** with a median latency of **138.39 ms**, providing an ultra-responsive experience well within the 200 ms interactive user threshold.
> - At **800x800**, the model delivers **5.82 FPS** with a median latency of **170.77 ms**, providing the superior detection accuracy on small litter objects while maintaining comfortable interactive speeds for photo analysis and web cam capture.

---

## 2. Pipeline Latency Breakdown (Stage by Stage)

### A. 640x640 Resolution
| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Std Dev (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Preprocessing** | 1.86 | 1.75 | 2.43 | 0.77 | 7.49 | 0.67 |
| **Model Inference** | 109.79 | 107.98 | 126.64 | 84.05 | 144.3 | 8.85 |
| **Postprocessing (NMS)** | 0.95 | 0.91 | 1.44 | 0.42 | 2.3 | 0.26 |
| **Total Wall-Clock** | **137.02** | **138.39** | **158.73** | **97.67** | **175.66** | **13.2** |

### B. 800x800 Resolution
| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Std Dev (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Preprocessing** | 4.38 | 4.39 | 5.15 | 1.99 | 5.86 | 0.62 |
| **Model Inference** | 163.51 | 162.63 | 177.78 | 121.07 | 236.17 | 12.83 |
| **Postprocessing (NMS)** | 0.93 | 0.94 | 1.23 | 0.4 | 1.39 | 0.18 |
| **Total Wall-Clock** | **171.93** | **170.77** | **186.22** | **128.78** | **244.68** | **12.99** |

---

## 3. Resource Utilization & Memory Footprint

- **Model Checkpoint File Size:** 18.3 MB
- **Model Load Time:** **128.23 ms**
- **Memory Footprint (RSS Delta on Load):** +46.59 MB
- **Process Peak RSS (640px run):** 518.0 MB
- **Process Peak RSS (800px run):** 593.96 MB
- **System Memory Safety Margin:** The 16 GB laptop has > 14 GB of headroom during CPU inference.

---

## 4. Hardware Verification & Safeguards Audit

1. **Production Checkpoint Unmodified:** [`models/weights_best.pt`](file:///d:/Projects/ml_project/472/models/weights_best.pt) was not accessed or written to.
2. **Production Code Unmodified:** Zero changes to `backend/`, `frontend/`, `api/`, or `launcher`.
3. **Protected Data Isolated:** `data/real_world_eval/` was never accessed or read.
4. **Benchmark Outputs Isolated:** All outputs written exclusively to `models/experiments/v5_taxonomy/training/cpu_benchmark/`.
