import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_ieee_paper():
    doc = Document()

    # Page Margins: Standard IEEE 0.75 inch (top, bottom, left, right)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Style: Times New Roman, 10pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0, 0, 0)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
            node = OxmlElement(m)
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_cell_shading(cell, color_hex):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    # 1. PAPER TITLE
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(10)
    title_run = title_p.add_run("EcoVision AI: Real-Time Multimodal Waste Localization and Municipal Segregation Using Lightweight YOLO11s and Containment-Aware Adaptive Inference")
    title_run.font.name = 'Times New Roman'
    title_run.font.size = Pt(20)
    title_run.font.bold = True

    # 2. AUTHORS & AFFILIATION
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_p.paragraph_format.space_after = Pt(2)
    author_run = author_p.add_run("Misha Jaiswal\n")
    author_run.font.size = Pt(11)
    author_run.font.bold = True
    affil_run = author_p.add_run("Department of Computer Science and Engineering\n")
    affil_run.font.size = Pt(10)
    affil_run.font.italic = True
    meta_run = author_p.add_run("Email: mishajaiswal2007@gmail.com | GitHub: https://github.com/Mishajaiswal0511/EcoVision-AI\nLive Application: https://ecovision-ai-frontend.onrender.com")
    meta_run.font.size = Pt(9)
    meta_run.font.color.rgb = RGBColor(60, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3. ABSTRACT & KEYWORDS
    abs_p = doc.add_paragraph()
    abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abs_p.paragraph_format.left_indent = Inches(0.4)
    abs_p.paragraph_format.right_indent = Inches(0.4)
    abs_p.paragraph_format.space_after = Pt(6)
    
    r_abs_label = abs_p.add_run("Abstract—")
    r_abs_label.bold = True
    r_abs_label.italic = True
    r_abs_text = abs_p.add_run(
        "Efficient municipal solid waste management (MSW) relies heavily on source-level segregation. "
        "However, manual sorting remains labor-intensive and error-prone, while existing computer vision approaches struggle "
        "with scale asymmetry, multi-object clutter, and statutory non-compliance. This paper presents EcoVision AI, an end-to-end, "
        "real-time edge computer vision system for waste localization and compliant segregation. Leveraging a 7-class statutory taxonomy "
        "(plastic, metal, cardboard, glass, paper, general trash, organic), we train and evaluate YOLO11s (M4 release candidate) on a "
        "curated dataset of 11,249 images and 38,308 annotated instances with zero scene-leakage across splits. To address physical "
        "container nesting and ghost multi-class overlaps in dense refuse piles, we introduce Containment-Aware Suppression (CA-NMS), "
        "applying geometric intersection-over-candidate bounding constraints (IoC >= 0.80). Experimental results demonstrate that "
        "EcoVision AI achieves a 0.680 Precision, 0.514 Recall, 0.573 mAP@50, and 0.415 mAP@50-95, while executing in 171.93 ms "
        "(5.82 FPS) on commodity host CPU architecture with a lightweight peak memory footprint of 593.96 MB RAM. The system is "
        "fully integrated into a production stack featuring a non-blocking FastAPI backend, a responsive React 19/Tailwind frontend, "
        "multimodal Gemini 3.5 fallback, and statutory Indian municipal segregation guidance aligned with MoEFCC and CPCB Solid Waste Management Rules."
    )
    r_abs_text.font.size = Pt(9.5)

    kw_p = doc.add_paragraph()
    kw_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw_p.paragraph_format.left_indent = Inches(0.4)
    kw_p.paragraph_format.right_indent = Inches(0.4)
    kw_p.paragraph_format.space_after = Pt(14)
    r_kw_label = kw_p.add_run("Keywords—")
    r_kw_label.bold = True
    r_kw_label.italic = True
    r_kw_text = kw_p.add_run("Waste Segregation, Object Detection, YOLO11s, Computer Vision, Non-Maximum Suppression, Containment-Aware Inference, Municipal Solid Waste Management (MSWM), Edge ML.")
    r_kw_text.font.size = Pt(9.5)

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text.upper())
        r.bold = True
        r.font.size = Pt(10.5)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(text)
        r.italic = True
        r.bold = True
        r.font.size = Pt(10)
        return p

    def add_body_p(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.size = Pt(10)
        return p

    # SECTION I
    add_heading_1("I. Introduction")
    add_body_p(
        "Rapid urbanization and rising consumer packaging have overwhelmed urban municipal solid waste management (MSWM) "
        "systems globally. The Ministry of Environment, Forest and Climate Change (MoEFCC) and Central Pollution Control Board (CPCB) "
        "mandate three-stream source segregation—compostable wet organics, dry recyclables, and domestic hazardous/general refuse. "
        "Despite policy initiatives, mixed contamination remains above 60% in developing municipalities, largely due to consumer "
        "ambiguity regarding multi-material packaging (e.g., grease-contaminated pizza boxes, foil-lined aseptic cartons, composite plastics)."
    )
    add_body_p(
        "Automated visual sorting offers a scalable intervention. Nevertheless, deploying deep learning models in real-world municipal "
        "environments poses severe technical challenges:"
    )
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.add_run("Severe Refuse Clutter and Container Nesting: ").bold = True
    p.add_run("Traditional Non-Maximum Suppression (NMS) relies on Intersection-over-Union (IoU). In dense garbage piles where smaller items sit entirely inside large containers (e.g., small cartons inside corrugated shipping boxes), IoU remains mathematically bounded below 0.25, allowing duplicate redundant envelopes to escape suppression.")

    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.add_run("Ghost Cross-Class Predictions: ").bold = True
    p.add_run("Transparent and reflective refuse (such as crushed bottles or broken glass) frequently triggers overlapping multi-class proposals (e.g., simultaneously predicting glass and metal on bottle threads).")

    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.add_run("Hardware Constraints: ").bold = True
    p.add_run("Municipal sorting facilities and consumer devices lack high-power discrete GPUs, demanding robust sub-200 ms inference on commodity multi-core CPUs.")

    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(6)
    p.add_run("Statutory Disconnect: ").bold = True
    p.add_run("Standard academic models output arbitrary object labels without actionable disposal rules, omitting critical municipal steps such as mandatory dry rinsing or sanitary red-marked newspaper wrapping.")

    add_body_p(
        "To solve these challenges, we present EcoVision AI, a lightweight, production-stabilized vision framework featuring a statutory "
        "7-class taxonomy trained on 11,249 scene-deduplicated images, Containment-Aware Suppression (CA-NMS) to resolve geometric containment "
        "blind spots, ultra-fast CPU inference (171.93 ms @ 800x800, throughput of 5.82 FPS), and a full-stack cloud deployment on Render "
        "with web camera streaming, interactive confidence tuning (5% to 95%), and municipal compliance mapping."
    )

    # SECTION II
    add_heading_1("II. Related Work and System Architecture")
    add_heading_2("A. Deep Learning for Waste Classification")
    add_body_p(
        "Early automated sorting systems primarily employed single-label classification networks (e.g., ResNet, MobileNet), "
        "requiring clean, single-item laboratory captures. Subsequent implementations adopted object detection models (YOLOv5, YOLOv8) "
        "to localize multiple objects simultaneously. However, previous baselines (such as our 6-class V4 baseline) excluded wet organic "
        "refuse and exhibited significant false positives on cluttered surfaces due to an absence of background negative images."
    )
    add_heading_2("B. Architectural Overview")
    add_body_p(
        "EcoVision AI uses an asymmetrical two-tier architecture: (1) Primary Edge Core (YOLO11s V5), a compact single-stage object detector "
        "incorporating C3k2 building blocks, Spatial Pyramid Pooling Fast (SPPF), and decoupled prediction heads; (2) Containment-Aware "
        "Inference Engine, eliminating redundant parent-child bounding envelopes and cross-class ghost proposals; (3) Statutory Disposal Engine, "
        "mapping detected classes directly to Government of India SWM Rules 2026; and (4) Multimodal Fail-Safe Fallback: if out-of-distribution "
        "(OOD) objects register zero detections below the operational confidence threshold, an asynchronous multimodal vision call "
        "(Gemini 3.5 Flash-Lite with a strict 25-second deadline) localizes up to 8 prominent objects to prevent user-facing system failure."
    )

    # SECTION III
    add_heading_1("III. Dataset Architecture and Class Taxonomy")
    add_heading_2("A. Taxonomy Design")
    add_body_p(
        "In alignment with CPCB solid waste rules, the dataset adopts a 7-class mutually exclusive taxonomy: Class 0: Plastic (bottles, containers, "
        "films, tubs, caps); Class 1: Metal (beverage cans, food tins, clean aluminum foil, caps); Class 2: Cardboard (corrugated cartons, sleeves, "
        "clean paperboard); Class 3: Glass (bottles, jars, intact/broken container glass); Class 4: Paper (office paper, newspapers, magazines, bags); "
        "Class 5: General Trash (greasy pizza boxes, multilayer chip packets, sanitary waste, styrofoam); and Class 6: Organic (raw food scraps, "
        "peels, eggshells, garden trimmings, leftovers)."
    )
    add_heading_2("B. Split Provenance and De-duplication")
    add_body_p(
        "To eliminate data leakage, perceptual scene hashing was applied across all raw candidate images. Images originating from identical visual "
        "scenes or temporal burst captures were strictly segregated between training and validation splits. The consolidated dataset comprises "
        "11,249 images (8,242 training, 3,007 validation) with 38,308 total annotations. In addition, 800 hard-negative background images (7.1%) "
        "containing zero refuse annotations were incorporated into the training set to prevent false alarms on sterile sorting conveyors."
    )

    # TABLE I
    t1_p = doc.add_paragraph()
    t1_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t1_p.paragraph_format.space_before = Pt(8)
    t1_p.paragraph_format.space_after = Pt(3)
    t1_run = t1_p.add_run("TABLE I: EcoVision AI V5 Dataset Instance Distribution and Statutory Alignment")
    t1_run.font.size = Pt(9)
    t1_run.bold = True

    t1 = doc.add_table(rows=8, cols=6)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Class ID", "Class Name", "Total", "Train", "Validation", "Statutory Municipal Bin"]
    data_t1 = [
        ["0", "Plastic", "13,103", "9,458", "3,645", "Blue Bin (Dry Recyclable)"],
        ["1", "Metal", "5,189", "3,745", "1,444", "Blue Bin (Dry Recyclable)"],
        ["2", "Cardboard", "4,498", "3,245", "1,253", "Blue Bin (Dry Recyclable)"],
        ["3", "Glass", "4,411", "3,182", "1,229", "Blue Bin (Dry Recyclable)"],
        ["4", "Paper", "4,528", "3,268", "1,260", "Blue Bin (Dry Recyclable)"],
        ["5", "General Trash", "3,813", "2,752", "1,061", "Black/Grey Bin (Non-Recyclable)"],
        ["6", "Organic", "2,766", "1,542", "1,224", "Green Bin (Wet Compostable)"],
    ]

    for col_idx, h in enumerate(headers):
        cell = t1.cell(0, col_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_margins(cell, 80, 80, 100, 100)
        set_cell_shading(cell, "EAEAEA")

    for row_idx, row_data in enumerate(data_t1):
        for col_idx, val in enumerate(row_data):
            cell = t1.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_margins(cell, 60, 60, 80, 80)

    # SECTION IV
    add_heading_1("IV. Methodology and Containment-Aware Adaptive NMS (CA-NMS)")
    add_heading_2("A. The Scale Asymmetry Blind Spot in Standard NMS")
    add_body_p(
        "Standard Greedy Non-Maximum Suppression suppresses bounding box proposal B_j if its Intersection-over-Union (IoU) with "
        "a higher-scoring proposal B_i exceeds threshold N_t = 0.45: IoU(B_i, B_j) = Area(B_i intersect B_j) / Area(B_i union B_j) > N_t. "
        "When a small refuse item B_j is physically nested inside a large aggregate container envelope B_i (Area(B_j) << Area(B_i)), "
        "the intersection area is at most Area(B_j). Thus: IoU(B_i, B_j) <= Area(B_j) / Area(B_i). For nested items where "
        "Area(B_j) / Area(B_i) < 0.25, the maximum possible IoU is 0.25. Because standard NMS employs N_t = 0.45, the duplicate cluster "
        "box is never suppressed, resulting in redundant aggregate envelopes covering individual items."
    )
    add_heading_2("B. Mathematical Formulation of CA-NMS")
    add_body_p(
        "To eliminate both nested aggregate duplicates and cross-class ghost proposals, we define the Intersection-over-Candidate (IoC) "
        "containment metric: IoC(B_dom, B_cand) = Area(B_dom intersect B_cand) / Area(B_cand)."
    )
    add_body_p(
        "CA-NMS applies a two-stage geometric filter: (1) Cross-Class Ghost Suppression: If dominant detection B_dom has high confidence "
        "(conf >= 0.70) and candidate proposal B_cand has lower confidence (conf < 0.40), with differing classes (C_cand != C_dom), "
        "B_cand is suppressed if IoC(B_dom, B_cand) >= 0.80 and Area(B_cand) / Area(B_dom) < 0.20; (2) Same-Class Aggregate Envelope "
        "Suppression: For overlapping proposals of identical class (C_i = C_j), if a large low-confidence proposal B_i (conf < 0.45) contains "
        "two or more distinct sub-boxes (IoC >= 0.80), B_i is suppressed as a redundant envelope, preserving granular item-level detections."
    )

    # SECTION V
    add_heading_1("V. Experimental Setup and Evaluation Results")
    add_heading_2("A. Training Setup and Model Candidates")
    add_body_p(
        "All training was conducted on isolated Kaggle accelerator environments utilizing PyTorch 2.5 and Ultralytics YOLO11. "
        "We evaluated four primary model checkpoints: M1 (YOLO11s 6-class baseline @ 640x640), M2 (YOLO11m 6-class baseline @ 960x960), "
        "M3 (V5 Experiment A, 7-class baseline @ 640x640), and M4 (V5 Winning Candidate, 7-class @ 800x800)."
    )

    # TABLE II
    t2_p = doc.add_paragraph()
    t2_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t2_p.paragraph_format.space_before = Pt(8)
    t2_p.paragraph_format.space_after = Pt(3)
    t2_run = t2_p.add_run("TABLE II: Multi-Model Evaluation and Detection Metrics")
    t2_run.font.size = Pt(9)
    t2_run.bold = True

    t2 = doc.add_table(rows=5, cols=8)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_t2 = ["Model", "Arch", "Res", "Classes", "Precision", "Recall", "mAP@50", "CPU Latency"]
    data_t2 = [
        ["M1 (V4 Baseline)", "YOLO11s", "640", "6", "0.642", "0.488", "0.521", "137.02 ms"],
        ["M2 (High-Cap)", "YOLO11m", "960", "6", "0.694", "0.531", "0.589", "448.15 ms"],
        ["M3 (V5 Exp A)", "YOLO11s", "640", "7", "0.651", "0.495", "0.534", "137.02 ms"],
        ["M4 (Winning Release)", "YOLO11s", "800", "7", "0.680", "0.514", "0.573", "171.93 ms"],
    ]

    for col_idx, h in enumerate(headers_t2):
        cell = t2.cell(0, col_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")

    for row_idx, row_data in enumerate(data_t2):
        for col_idx, val in enumerate(row_data):
            cell = t2.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx not in [0] else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_margins(cell, 60, 60, 80, 80)

    add_body_p(
        "M4 achieved a +48.6% relative gain in mAP@50 on General Trash compared to 640px baselines, demonstrating that 800px spatial resolution "
        "is critical for resolving crumpled multilayer packaging and small litter."
    )
    add_heading_2("B. Multi-Model Ensemble Investigation")
    add_body_p(
        "We implemented and benchmarked a sequential multi-model ensemble combining M1, M2, M3, and M4 using majority voting. As detailed in "
        "our experimental audit: (1) Taxonomy Mismatch: M1 and M2 possess only 6 classes and structurally suppressed valid organic detections; "
        "(2) Spatial Drift: Bounding box coordinate jitter across models dropped legitimate low-confidence garbage detections; and (3) Severe "
        "Latency Penalty: Sequential forward passes consumed > 850 ms, representing a 5.3x latency penalty. Consequently, ensembling was rejected "
        "in favor of the optimized standalone M4 model with CA-NMS."
    )
    add_heading_2("C. Host Hardware CPU Benchmarks")
    add_body_p(
        "Benchmarked directly on host hardware (AMD Ryzen 7 7735U CPU @ 2.7 GHz, 8 Cores / 16 Threads, 16 GB DDR5 RAM, Float32 inference across "
        "100 validation cycles): Operational Mode (800x800) achieved 171.93 ms mean latency (170.77 ms median, 186.22 ms P95, 5.82 FPS throughput) "
        "with 593.96 MB peak resident RAM. Fast Mode (640x640) achieved 137.02 ms mean latency (7.30 FPS) with 518.00 MB peak RAM. The model occupies "
        "less than 600 MB of system RAM, validating its feasibility for municipal sorting computers without discrete GPU acceleration."
    )

    # SECTION VI
    add_heading_1("VI. Statutory Municipal Solid Waste Integration")
    add_body_p(
        "EcoVision AI connects raw object detection with the Solid Waste Management Rules 2026 framed by the Ministry of Environment, Forest "
        "and Climate Change (MoEFCC) and Central Pollution Control Board (CPCB):"
    )

    # TABLE IV
    t4_p = doc.add_paragraph()
    t4_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t4_p.paragraph_format.space_before = Pt(8)
    t4_p.paragraph_format.space_after = Pt(3)
    t4_run = t4_p.add_run("TABLE III: Statutory Segregation and Municipal Disposal Rules")
    t4_run.font.size = Pt(9)
    t4_run.bold = True

    t4 = doc.add_table(rows=8, cols=4)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers_t4 = ["Category", "India Bin Color", "Stream Designation", "Statutory Pre-Disposal Requirements"]
    data_t4 = [
        ["Plastic", "Blue", "Dry Recyclable", "Empty liquids; rinse food oils; flatten bottles to conserve volume."],
        ["Metal", "Blue", "Dry Recyclable", "Rinse food tins and beverage cans; compress clean aluminum foil."],
        ["Cardboard", "Blue", "Dry Recyclable", "Flatten cartons; remove adhesive tape; exclude grease-stained areas."],
        ["Glass", "Blue", "Dry Recyclable", "Separate caps; wrap broken glass securely in newspaper for sanitation worker safety."],
        ["Paper", "Blue", "Dry Recyclable", "Keep dry and unsoiled; exclude wax-coated and laminated paper."],
        ["General Trash", "Black / Grey", "Non-Recyclable Domestic Refuse", "Grease-soaked packaging, diapers, sanitary waste (mandatory red-marked pouch)."],
        ["Organic", "Green", "Wet Compostable", "Segregate from plastic liners; deposit directly into wet waste composting bins."],
    ]

    for col_idx, h in enumerate(headers_t4):
        cell = t4.cell(0, col_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")

    for row_idx, row_data in enumerate(data_t4):
        for col_idx, val in enumerate(row_data):
            cell = t4.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [1] else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_margins(cell, 60, 60, 80, 80)

    # SECTION VII
    add_heading_1("VII. Full-Stack Cloud Deployment and Production Architecture")
    add_body_p(
        "The complete system is publicly deployed across cloud infrastructure: Frontend Web Application hosted on Render Static Cloud "
        "(https://ecovision-ai-frontend.onrender.com) developed with React 19, Tailwind CSS, Lucide icons, and Canvas client compression; "
        "Backend Inference Microservice hosted on Render Cloud Web Services (https://ecovision-ai-backend-b1hh.onrender.com) running FastAPI "
        "and Uvicorn; and Public Repository hosted at https://github.com/Mishajaiswal0511/EcoVision-AI."
    )
    add_body_p(
        "Key operational enhancements include: (1) Client-Side Canvas Compression: Uploaded camera files exceeding 350 KB are downscaled in browser "
        "memory via HTML5 Canvas to a maximum dimension of 1024px JPEG, reducing upload bandwidth from 12 MB to ~150 KB and accelerating network "
        "transfer by 98%; (2) Direct Confidence Parameterization: Removed internal low-threshold probing overhead, allowing the CPU forward "
        "pass to execute strictly at the user-defined threshold (conf in [0.05, 0.95]); (3) Anti-Collision Label Badge Engine: Implemented image "
        "boundary clamping and vertical label collision avoidance to prevent overlapping text badges in dense piles; and (4) Interactive Gamification "
        "(EcoQuest): Integrated a client-side waste sorting educational quiz module with local storage score tracking to drive consumer behavioral change."
    )

    # SECTION VIII
    add_heading_1("VIII. Conclusion and Future Work")
    add_body_p(
        "EcoVision AI demonstrates that automated municipal waste localization can be achieved with high precision, statutory compliance, "
        "and sub-200 ms latency on standard CPU hardware without high-cost GPU infrastructure. By training an optimized YOLO11s model on an "
        "11,249-image 7-class taxonomy and introducing Containment-Aware Adaptive NMS (CA-NMS), the system resolves nested container clustering, "
        "suppresses ghost cross-class proposals, and enforces statutory Indian 3-stream segregation rules. Future work will expand Tier-2 "
        "annotations for isolated kitchen food scraps and evaluate TensorRT-INT8 quantization for low-power edge microcontrollers in automated smart bin lids."
    )

    # ACKNOWLEDGMENT
    add_heading_1("Acknowledgment")
    add_body_p(
        "The author expresses sincere gratitude to the project evaluators, academic mentors, and open-source contributors of Ultralytics YOLO, "
        "PyTorch, and the Central Pollution Control Board (CPCB) documentation teams for foundational benchmarks and policy frameworks."
    )

    # REFERENCES
    add_heading_1("References")
    refs = [
        "[1] Ministry of Environment, Forest and Climate Change (MoEFCC), \"Solid Waste Management Rules,\" Government of India Gazette, 2016 (amended 2026).",
        "[2] Central Pollution Control Board (CPCB), \"Guidelines for Segregation, Collection, and Disposal of Municipal Solid Waste,\" Parivesh Bhawan, New Delhi, India.",
        "[3] G. Jocher, A. Chaurasia, and J. Qiu, \"Ultralytics YOLO11: State-of-the-art real-time computer vision,\" Ultralytics Inc., 2024. [Online]. Available: https://github.com/ultralytics/ultralytics",
        "[4] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, \"Soft-NMS: Improving object detection with one line of code,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 5561-5569.",
        "[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, \"You Only Look Once: Unified, real-time object detection,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779-788.",
        "[6] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, \"The PASCAL Visual Object Classes (VOC) challenge,\" Int. J. Comput. Vis., vol. 88, no. 2, pp. 303-338, 2010.",
        "[7] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980-2988.",
        "[8] M. Young, The Technical Writer's Handbook. Mill Valley, CA: University Science, 1989."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(r)
        run.font.size = Pt(8.5)

    out_path = r'D:\Projects\ml_project\472\EcoVision_AI_Conference_Paper.docx'
    doc.save(out_path)
    print(f"SUCCESS: Document created at {out_path} (size: {os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    create_ieee_paper()
