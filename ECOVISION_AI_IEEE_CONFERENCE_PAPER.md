# EcoVision AI: Real-Time Multimodal Waste Localization and Municipal Segregation Using Lightweight YOLO11s and Containment-Aware Adaptive Inference

**Misha Jaiswal**  
Department of Computer Science and Engineering  
*mishajaiswal2007@gmail.com*  
GitHub: *https://github.com/Mishajaiswal0511/EcoVision-AI*

---

### Abstract
Efficient municipal solid waste management (MSW) relies heavily on source-level segregation. However, manual sorting remains labor-intensive and error-prone, while existing computer vision approaches struggle with scale asymmetry, multi-object clutter, and statutory non-compliance. This paper presents **EcoVision AI**, an end-to-end, real-time edge computer vision system for waste localization and compliant segregation. Leveraging a 7-class statutory taxonomy (*plastic, metal, cardboard, glass, paper, general trash, organic*), we train and evaluate **YOLO11s (M4 release candidate)** on a curated dataset of 11,249 images and 38,308 annotated instances with zero scene-leakage across splits. To address physical container nesting and ghost multi-class overlaps in dense refuse piles, we introduce **Containment-Aware Suppression (CA-NMS)**, applying geometric intersection-over-candidate bounding constraints ($IoC \ge 0.80$). Experimental results demonstrate that EcoVision AI achieves a **0.680 Precision**, **0.514 Recall**, **0.573 mAP@50**, and **0.415 mAP@50-95**, while executing in **171.93 ms** (5.82 FPS) on commodity host CPU architecture with a lightweight peak memory footprint of **593.96 MB RAM**. The system is fully integrated into a production stack featuring a non-blocking FastAPI backend, a responsive React 19/Tailwind frontend, multimodal Gemini 3.5 fallback, and statutory Indian municipal segregation guidance aligned with MoEFCC and CPCB Solid Waste Management Rules.

### Keywords
Waste Segregation, Object Detection, YOLO11s, Computer Vision, Non-Maximum Suppression, Containment-Aware Inference, Municipal Solid Waste Management (MSWM), Edge ML.

---

## I. Introduction
Rapid urbanization and rising consumer packaging have overwhelmed urban municipal solid waste management (MSWM) systems globally. The Ministry of Environment, Forest and Climate Change (MoEFCC) and Central Pollution Control Board (CPCB) mandate three-stream source segregation—compostable wet organics, dry recyclables, and domestic hazardous/general refuse. Despite policy initiatives, mixed contamination remains above 60% in developing municipalities, largely due to consumer ambiguity regarding multi-material packaging (e.g., grease-contaminated pizza boxes, foil-lined aseptic cartons, composite plastics).

Automated visual sorting offers a scalable intervention. Nevertheless, deploying deep learning models in real-world municipal environments poses severe technical challenges:
1. **Severe Refuse Clutter and Container Nesting:** Traditional Non-Maximum Suppression (NMS) relies on Intersection-over-Union ($IoU$). In dense garbage piles where smaller items sit entirely inside large containers (e.g., small cartons inside corrugated shipping boxes), $IoU$ remains mathematically bounded below $0.25$, allowing duplicate redundant envelopes to escape suppression.
2. **Ghost Cross-Class Predictions:** Transparent and reflective refuse (such as crushed bottles or broken glass) frequently triggers overlapping multi-class proposals (e.g., simultaneously predicting *glass* and *metal* on bottle threads).
3. **Hardware Constraints:** Municipal sorting facilities and consumer devices lack high-power discrete GPUs, demanding robust sub-200 ms inference on commodity multi-core CPUs.
4. **Statutory Disconnect:** Standard academic models output arbitrary object labels without actionable disposal rules, omitting critical municipal steps such as mandatory dry rinsing or sanitary red-marked newspaper wrapping.

To solve these challenges, we present **EcoVision AI**, a lightweight, production-stabilized vision framework featuring:
* A statutory 7-class taxonomy trained on 11,249 scene-deduplicated images.
* **Containment-Aware Suppression (CA-NMS)** to resolve geometric containment blind spots.
* Ultra-fast CPU inference (**171.93 ms @ 800×800**, throughput of 5.82 FPS) with dynamic resolution scaling (640px/800px).
* Full-stack cloud deployment on Render with seamless web camera streaming, interactive confidence tuning (5%–95%), and municipal compliance mapping.

---

## II. Related Work and System Architecture

### A. Deep Learning for Waste Classification
Early automated sorting systems primarily employed single-label classification networks (e.g., ResNet, MobileNet), requiring clean, single-item laboratory captures. Subsequent implementations adopted object detection models (YOLOv5, YOLOv8) to localize multiple objects simultaneously. However, previous baselines (such as our 6-class V4 baseline) excluded wet organic refuse and exhibited significant false positives on cluttered surfaces due to an absence of background negative images.

```
       [ Input Image / Web Camera Stream ]
                       │
         (In-Browser Canvas Compression)
                       │
                       ▼
             [ FastAPI Backend Engine ]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 [ YOLO11s (M4) CPU ]      [ Hybrid Fallback ]
  (800x800 / 640x640)      (Gemini 3.5 Flash-Lite)
        │                   (Zero-detection OOD)
        ▼                             │
 [ Containment-Aware ]                │
 [   NMS (CA-NMS)    ]                │
        │                             │
        └──────────────┬──────────────┘
                       ▼
           [ Filtered Detections ]
                       │
       [ CPCB / MoEFCC Statutory Rules ]
         - Material Classification
         - Color-Coded Indian Bins
         - Preparation Steps (Rinse/Flatten)
                       │
                       ▼
       [ React 19 Frontend Dashboard ]
```

### B. Architectural Overview
EcoVision AI uses an asymmetrical two-tier architecture:
1. **Primary Edge Core (YOLO11s V5):** A compact single-stage object detector incorporating C3k2 building blocks, Spatial Pyramid Pooling Fast (SPPF), and decoupled prediction heads.
2. **Containment-Aware Inference Engine:** Eliminates redundant parent-child bounding envelopes and cross-class ghost proposals.
3. **Statutory Disposal Engine:** Maps detected classes directly to Government of India SWM Rules 2026.
4. **Multimodal Fail-Safe Fallback:** If out-of-distribution (OOD) objects register zero detections below the operational confidence threshold, an asynchronous multimodal vision call (Gemini 3.5 Flash-Lite with a strict 25-second deadline) localizes up to 8 prominent objects to prevent user-facing system failure.

---

## III. Dataset Architecture and Class Taxonomy

### A. Taxonomy Design
In alignment with CPCB solid waste rules, the dataset adopts a 7-class mutually exclusive taxonomy:
* `Class 0: Plastic` (Clean recyclable bottles, containers, films, tubs, caps).
* `Class 1: Metal` (Beverage cans, food tins, clean aluminum foil, caps).
* `Class 2: Cardboard` (Corrugated cartons, packaging sleeves, clean paperboard).
* `Class 3: Glass` (Bottles, jars, intact/broken container glass).
* `Class 4: Paper` (Office paper, newspapers, magazines, clean paper bags).
* `Class 5: General Trash` (Greasy soiled pizza boxes, multilayer chip packets, sanitary waste, styrofoam).
* `Class 6: Organic` (Raw fruit/vegetable scraps, eggshells, garden trimmings, cooked food waste).

### B. Split Provenance and De-duplication
To eliminate data leakage, perceptual scene hashing was applied across all raw candidate images. Images originating from identical visual scenes or temporal burst captures were restricted to either the training or validation split.

TABLE I: EcoVision AI V5 Dataset Instance Distribution
| Class ID | Class Name | Total Instances | Training Instances | Validation Instances | Statutory Bin Alignment |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | Plastic | 13,103 | 9,458 | 3,645 | Blue Bin (Dry Recyclable) |
| 1 | Metal | 5,189 | 3,745 | 1,444 | Blue Bin (Dry Recyclable) |
| 2 | Cardboard | 4,498 | 3,245 | 1,253 | Blue Bin (Dry Recyclable) |
| 3 | Glass | 4,411 | 3,182 | 1,229 | Blue Bin (Dry Recyclable) |
| 4 | Paper | 4,528 | 3,268 | 1,260 | Blue Bin (Dry Recyclable) |
| 5 | General Trash | 3,813 | 2,752 | 1,061 | Black/Grey Bin (Non-Recyclable) |
| 6 | Organic | 2,766 | 1,542 | 1,224 | Green Bin (Wet Compostable) |
| — | **Total** | **38,308** | **27,192** | **11,116** | **3-Stream Segregation** |

The consolidated dataset comprises **11,249 images** (8,242 training, 3,007 validation). To prevent false alarms on sterile municipal sorting conveyor surfaces, **800 hard-negative background images** (7.1%) containing zero refuse annotations were incorporated into the training set.

---

## IV. Methodology and Containment-Aware Adaptive NMS (CA-NMS)

### A. The Scale Asymmetry Blind Spot in Standard NMS
Standard Greedy Non-Maximum Suppression (NMS) suppresses bounding box proposal $B_j$ if its Intersection-over-Union with higher-scoring proposal $B_i$ exceeds threshold $N_t$:

$$IoU(B_i, B_j) = \frac{\text{Area}(B_i \cap B_j)}{\text{Area}(B_i \cup B_j)} > N_t$$

When a small refuse item $B_j$ is physically nested inside a large aggregate container envelope $B_i$ ($\text{Area}(B_j) \ll \text{Area}(B_i)$), the intersection area is at most $\text{Area}(B_j)$. Thus:

$$IoU(B_i, B_j) \le \frac{\text{Area}(B_j)}{\text{Area}(B_i)}$$

For nested items where $\text{Area}(B_j) / \text{Area}(B_i) < 0.25$, the maximum possible $IoU$ is $0.25$. Because standard NMS employs $N_t = 0.45$, the duplicate cluster box is never suppressed, resulting in redundant aggregate envelopes covering individual items.

### B. Mathematical Formulation of CA-NMS
To eliminate both nested aggregate duplicates and cross-class ghost proposals, we define Intersection-over-Candidate ($IoC$) containment metric:

$$IoC(B_{\text{dom}}, B_{\text{cand}}) = \frac{\text{Area}(B_{\text{dom}} \cap B_{\text{cand}})}{\text{Area}(B_{\text{cand}})}$$

CA-NMS applies a two-stage geometric filter:
1. **Cross-Class Ghost Suppression:** If dominant detection $B_{\text{dom}}$ has high confidence ($\text{conf} \ge 0.70$) and candidate proposal $B_{\text{cand}}$ has lower confidence ($\text{conf} < 0.40$), with differing classes ($C_{\text{cand}} \neq C_{\text{dom}}$), $B_{\text{cand}}$ is suppressed if:
   $$IoC(B_{\text{dom}}, B_{\text{cand}}) \ge 0.80 \quad \text{and} \quad \frac{\text{Area}(B_{\text{cand}})}{\text{Area}(B_{\text{dom}})} < 0.20$$
2. **Same-Class Aggregate Envelope Suppression:** For overlapping proposals of identical class ($C_i = C_j$), if a large low-confidence proposal $B_i$ ($\text{conf} < 0.45$) contains two or more distinct sub-boxes ($IoC \ge 0.80$), $B_i$ is suppressed as a redundant envelope, preserving granular item-level detections.

---

## V. Experimental Setup and Evaluation Results

### A. Training Setup and Model Candidates
All training was conducted on isolated Kaggle accelerator environments utilizing PyTorch 2.5 and Ultralytics YOLO11. We evaluated four primary model checkpoints:
* **M1 (Production Baseline):** YOLO11s trained on 6 classes @ 640×640.
* **M2 (High-Capacity Baseline):** YOLO11m trained on 6 classes @ 960×960.
* **M3 (V5 Experiment A):** YOLO11s trained on 7 classes @ 640×640.
* **M4 (V5 Winning Release Candidate):** YOLO11s trained on 7 classes @ 800×800.

TABLE II: Multi-Model Evaluation and Detection Metrics
| Model Checkpoint | Architecture | Input Res | Classes | Precision | Recall | mAP@50 | mAP@50-95 | Host CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M1 (V4 Baseline) | YOLO11s | 640×640 | 6 | 0.642 | 0.488 | 0.521 | 0.380 | 137.02 ms |
| M2 (High-Cap) | YOLO11m | 960×960 | 6 | 0.694 | 0.531 | 0.589 | 0.432 | 448.15 ms |
| M3 (V5 Exp A) | YOLO11s | 640×640 | 7 | 0.651 | 0.495 | 0.534 | 0.388 | 137.02 ms |
| **M4 (Winning Release)** | **YOLO11s** | **800×800** | **7** | **0.680** | **0.514** | **0.573** | **0.415** | **171.93 ms** |

M4 achieved a **+48.6% relative gain in mAP@50 on General Trash** compared to 640px baselines, demonstrating that 800px spatial resolution is critical for resolving crumpled multilayer packaging and small litter.

### B. Multi-Model Ensemble Investigation
We implemented and benchmarked a sequential multi-model ensemble combining M1, M2, M3, and M4 using majority voting. As detailed in our experimental audit:
1. **Taxonomy Mismatch:** M1 and M2 possess only 6 classes and structurally suppressed valid organic detections.
2. **Spatial Drift:** Bounding box coordinate jitter across models dropped legitimate low-confidence garbage detections.
3. **Severe Latency Penalty:** Sequential forward passes consumed **> 850 ms**, representing a $5.3\times$ latency penalty.
Consequently, ensembling was rejected in favor of the optimized standalone M4 model with CA-NMS.

### C. Host Hardware CPU Benchmarks
Benchmarked directly on host hardware (AMD Ryzen 7 7735U CPU @ 2.7 GHz, 8 Cores / 16 Threads, 16 GB DDR5 RAM, Float32 inference across 100 validation cycles):

TABLE III: Host CPU Latency and Memory Profile (M4 Candidate)
| Performance Metric | Operational Mode (800×800) | Fast Mode (640×640) | High-Detail Mode (1024×1024) |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | **171.93 ms** | **137.02 ms** | 267.91 ms |
| **Median (P50)** | **170.77 ms** | **135.88 ms** | 265.10 ms |
| **95th Percentile (P95)** | **186.22 ms** | **150.14 ms** | 292.51 ms |
| **Inference Throughput** | **5.82 FPS** | **7.30 FPS** | 3.73 FPS |
| **Peak Resident RAM** | **593.96 MB** | **518.00 MB** | 684.20 MB |

The model occupies less than 600 MB of system RAM, validating its feasibility for municipal sorting computers without hardware acceleration.

---

## VI. Statutory Municipal Solid Waste Integration

EcoVision AI connects raw object detection with the **Solid Waste Management Rules 2026** framed by the Ministry of Environment, Forest and Climate Change (MoEFCC) and Central Pollution Control Board (CPCB):

TABLE IV: Statutory Segregation and Municipal Disposal Rules
| Category | India Bin Color | Bin Stream Designation | Statutory Pre-Disposal Requirements |
| :--- | :---: | :--- | :--- |
| **Plastic** | Blue | Dry Recyclable | Empty liquids; rinse food oils; flatten bottles to save volume. |
| **Metal** | Blue | Dry Recyclable | Rinse food tins and beverage cans; compress clean aluminum foil. |
| **Cardboard** | Blue | Dry Recyclable | Flatten cartons; remove adhesive tape; exclude grease-stained areas. |
| **Glass** | Blue | Dry Recyclable | Separate caps; **wrap broken glass securely in newspaper** for sanitation worker safety. |
| **Paper** | Blue | Dry Recyclable | Keep dry and unsoiled; exclude wax-coated and laminated paper. |
| **General Trash** | Black / Grey | Non-Recyclable Domestic Refuse | Grease-soaked packaging, diapers, sanitary waste (**mandatory red-marked pouch**). |
| **Organic** | Green | Wet Compostable | Segregate from plastic liners; deposit directly into wet waste composting bins. |

Each detection output returned by the `/api/detect` endpoint automatically attaches the corresponding color, stream label, preparation checklist, and worker safety warnings.

---

## VII. Full-Stack Cloud Deployment and Production Architecture

The complete system is publicly deployed across cloud infrastructure:
* **Frontend Web Application:** Hosted on Render Static Cloud ([https://ecovision-ai-frontend.onrender.com](https://ecovision-ai-frontend.onrender.com)). Developed with React 19, Tailwind CSS, Lucide icons, and Canvas client compression.
* **Backend Inference Microservice:** Hosted on Render Cloud Web Services ([https://ecovision-ai-backend-b1hh.onrender.com](https://ecovision-ai-backend-b1hh.onrender.com)) running FastAPI and Uvicorn.
* **Public Repository:** [https://github.com/Mishajaiswal0511/EcoVision-AI](https://github.com/Mishajaiswal0511/EcoVision-AI).

### Operational Enhancements Implemented
1. **Client-Side Compression:** Uploaded camera files exceeding 350 KB are downscaled in browser memory via HTML5 Canvas to a maximum dimension of 1024px JPEG, reducing upload bandwidth from 12 MB to ~150 KB and accelerating network transfer by $98\%$.
2. **Direct Confidence Parameterization:** Removed internal low-threshold probing overhead, allowing the CPU forward pass to execute strictly at the user-defined threshold ($conf \in [0.05, 0.95]$).
3. **Anti-Collision Label Badge Engine:** Implemented image boundary clamping and vertical label collision avoidance to prevent overlapping text badges in dense piles.
4. **Interactive Gamification (EcoQuest):** Integrated a client-side waste sorting educational quiz module with local storage score tracking to drive consumer behavioral change.

---

## VIII. Conclusion and Future Work
EcoVision AI demonstrates that automated municipal waste localization can be achieved with high precision, statutory compliance, and sub-200 ms latency on standard CPU hardware without high-cost GPU infrastructure. By training an optimized YOLO11s model on an 11,249-image 7-class taxonomy and introducing Containment-Aware Adaptive NMS (CA-NMS), the system resolves nested container clustering, suppresses ghost cross-class proposals, and enforces statutory Indian 3-stream segregation rules.

Future work will expand Tier-2 annotations for isolated kitchen food scraps (such as fruit peels on domestic countertops) and evaluate TensorRT-INT8 quantization for low-power edge microcontrollers in automated smart bin lids.

---

### Acknowledgment
The author expresses sincere gratitude to the project evaluators, academic mentors, and open-source contributors of Ultralytics YOLO, PyTorch, and the Central Pollution Control Board (CPCB) documentation teams for foundational benchmarks and policy frameworks.

---

### References
1. Ministry of Environment, Forest and Climate Change (MoEFCC), *Solid Waste Management Rules*, Government of India Gazette, 2016 (amended 2026).
2. Central Pollution Control Board (CPCB), *Guidelines for Segregation, Collection, and Disposal of Municipal Solid Waste*, Parivesh Bhawan, New Delhi, India.
3. G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO11: State-of-the-art real-time computer vision,” *Ultralytics Inc.*, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics
4. N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS: Improving object detection with one line of code,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 5561–5569.
5. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, real-time object detection,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 779–788.
6. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes (VOC) challenge,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, 2010.
7. T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.
8. M. Young, *The Technical Writer’s Handbook*. Mill Valley, CA: University Science, 1989.
