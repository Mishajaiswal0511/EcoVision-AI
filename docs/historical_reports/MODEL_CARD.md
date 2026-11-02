# EcoVision AI — Model Card: YOLO11s V5 (Candidate M4)

## Model Overview
* **Model Name:** EcoVision AI YOLO11s V5 Candidate (M4)
* **Architecture:** Ultralytics YOLO11s (Single-Stage CNN Object Detector)
* **Model Size:** 9.4 Million Parameters (19.2 MB `.pt` file)
* **Input Resolution:** $800\times 800$ pixels (Standard), $1024\times 1024$ pixels (High-Detail)
* **Target Inference Hardware:** CPU (x86_64, AMD Ryzen 7 7735U / Intel Core i5/i7)
* **Checkpoint Path:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`
* **SHA-256 Digest:** `bf3890b8374bfae6c376a898fc2080f79095aae0e9bd37ed612e86fb1dffc66a`
* **Governing Policy:** MoEFCC / CPCB Solid Waste Management Rules 2026 (Government of India)

---

## Intended Use
* **Primary Objective:** Real-time localization and categorization of household and commercial waste items from camera streams or uploaded images to guide municipal waste segregation.
* **Intended Users:** Waste management operators, academic researchers, municipal facility supervisors, and citizen users of recycling drop-off kiosks.
* **Deployment Context:** Local desktop or edge web application running on CPU without dedicated GPU acceleration.

---

## Class Taxonomy & Color Conventions

| Class ID | Class Name | Target Category | Municipal Bin Label | India Bin Color | Recyclable? |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **0** | `plastic` | Rigid containers, PET bottles, clean tubs | Blue Bin (Dry Recyclables) | Blue | YES |
| **1** | `metal` | Aluminum cans, food tins, scrap foil | Blue Bin (Dry Recyclables) | Blue / Amber | YES |
| **2** | `cardboard` | Corrugated boxes, packaging cartons | Blue Bin (Dry Recyclables) | Blue / Brown | YES |
| **3** | `glass` | Glass beverage bottles, food jars | Blue Bin / Glass Bank | Blue / Teal | YES (Wrap if broken) |
| **4** | `paper` | Office sheets, clean paper bags | Blue Bin (Dry Recyclables) | Blue / Emerald | YES |
| **5** | `general_trash` | Metallized packets, foam, composite litter | Black/Grey Bin (Residual Waste) | Black / Grey | NO (Landfill/RDF) |
| **6** | `organic` | Kitchen food scraps, peels, vegetable waste| Green Bin (Wet Waste) | Green | NO (Compostable) |

---

## Training Data & Provenance
* **Dataset:** EcoVision AI V5 (`data/processed_v5/`)
* **Total Images:** 11,249 (8,242 train, 3,007 validation)
* **Total Instances:** 38,308 bounding boxes
* **Data Sources:** Curated, cleaned, and scene-deduplicated subsets of:
  - TACO (Trash Annotations in Context)
  - Garbage in the Wild (GWD)
  - Roboflow Public Waste repositories
* **Scene Separation:** Scene-hash grouped; strict 0% train/validation parent scene overlap.
* **Negative Backgrounds:** 800 verified empty background images ($7.1\%$) to suppress false alarms on clean countertops.

---

## Performance Metrics

### Algorithmic Metrics (3,007 Validation Images):
* **Precision:** **0.680** (68.0%)
* **Recall:** **0.514** (51.4%)
* **mAP@50 (In-Training, `conf=0.001`):** **0.573** (57.3%)
* **mAP@50-95:** **0.415** (41.5%)
* **mAP@50 (Operational, `conf=0.15`):** **0.520** (52.0%)

### CPU Latency Benchmark (AMD Ryzen 7 7735U, 16 GB RAM):
* **Model Load Time:** **0.177 seconds**
* **Operational Latency (@ 800px):**
  - Mean: **183.20 ms**
  - Median (P50): **180.50 ms**
  - 95th Percentile (P95): **220.85 ms**
  - Throughput: **5.46 FPS**
* **High-Detail Latency (@ 1024px):**
  - Mean: **267.91 ms**
  - P95: **292.51 ms**
  - Throughput: **3.73 FPS**
* **Memory Footprint:** Peak RSS RAM **~594 MB** ($< 4\%$ of 16 GB RAM)

---

## Known Limitations & Out-of-Scope Use
1. **Standalone Fruit Peels (Banana Peel Anomaly):**
   - The training set contains clustered compost tray scraps but zero standalone banana peels on clean indoor tables.
   - The model can misclassify single banana peels as `glass` (~60% conf) or `plastic` due to specular highlights.
2. **General Trash Recall:**
   - Because of extreme visual variance (foil bags, cigarette butts, Styrofoam), General Trash recall is $14.3\%$.
3. **Food-Contact Grease Ambiguity:**
   - The detector cannot detect chemical oils. Greasy pizza boxes must be segregated conditionally using the application's guidance.
4. **Out-of-Scope:**
   - Hazardous industrial chemicals, biological hospital waste, and high-voltage electronics.

