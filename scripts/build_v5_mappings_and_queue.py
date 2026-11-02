"""
EcoVision AI - V5 Mapping Review & Manual Review Queue Builder
Generates mapping_review.csv and manual_review_queue.csv based on MoEFCC/CPCB 2026 rules.
"""

import csv
import json
from pathlib import Path

OUTPUT_DIR = Path("models/experiments/v5_taxonomy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate mapping_review.csv
# Columns: source_category, source_image_count, source_instance_count, current_class, proposed_v5_class, mapping_type, review_required, reason, confidence

MAPPING_REVIEW_ROWS = [
    # Plastic packaging
    ["Clear plastic bottle", 484, 484, "plastic", "plastic", "DIRECT", "NO", "Clean recyclable PET beverage container accepted in Blue Bin", "HIGH"],
    ["Other plastic bottle", 89, 89, "plastic", "plastic", "DIRECT", "NO", "Clean recyclable HDPE/PP cleaning and cosmetic bottles", "HIGH"],
    ["Disposable plastic cup", 176, 176, "plastic", "plastic", "DIRECT", "NO", "Rigid clean plastic drinking cups accepted for recycling", "HIGH"],
    ["Plastic lid", 138, 138, "plastic", "plastic", "DIRECT", "NO", "Clean plastic container/cup lids", "HIGH"],
    ["Plastic bottle cap", 348, 348, "plastic", "plastic", "DIRECT", "NO", "Clean plastic screw caps accepted with bottles", "HIGH"],
    ["Spread tub", 17, 17, "plastic", "plastic", "DIRECT", "NO", "Clean butter/margarine plastic tubs", "HIGH"],
    ["Tupperware", 7, 7, "plastic", "plastic", "DIRECT", "NO", "Rigid reusable plastic containers", "HIGH"],
    ["Other plastic container", 9, 9, "plastic", "plastic", "DIRECT", "NO", "Clean rigid plastic tubs and trays", "HIGH"],
    ["Other plastic cup", 4, 4, "plastic", "plastic", "DIRECT", "NO", "Plastic drinking cups", "HIGH"],
    ["Other plastic", 456, 456, "plastic", "plastic", "DIRECT", "NO", "Rigid plastic items and clean packaging", "MEDIUM"],
    ["Plastic film", 789, 789, "plastic", "plastic", "DIRECT", "NO", "Clean polyethylene film/wrap accepted in dry recycling", "HIGH"],
    ["Single-use carrier bag", 104, 104, "plastic", "plastic", "DIRECT", "NO", "Clean polyethylene plastic carry bags", "HIGH"],
    ["Polypropylene bag", 6, 6, "plastic", "plastic", "DIRECT", "NO", "Clean woven PP sacks and bags", "HIGH"],
    ["Six pack rings", 21, 21, "plastic", "plastic", "DIRECT", "NO", "LDPE can rings", "HIGH"],
    ["Garbage bag", 52, 52, "plastic", "plastic", "CONDITIONAL", "YES", "Empty plastic film bags are recyclable; full bags are residual waste containers", "MEDIUM"],
    ["Other plastic wrapper", 446, 446, "plastic", "plastic", "CONDITIONAL", "YES", "Clean film wrappers are plastic; greasy food wrappers require residual disposal", "MEDIUM"],
    # Non-recyclable plastics -> General Trash
    ["Crisp packet", 66, 66, "plastic", "general_trash", "DIRECT", "NO", "Metallized BOPP multi-layer film is non-recyclable curbside under MoEFCC rules", "HIGH"],
    ["Foam cup", 24, 24, "plastic", "general_trash", "DIRECT", "NO", "Expanded polystyrene (EPS) is non-recyclable residual waste under MoEFCC rules", "HIGH"],
    ["Foam food container", 26, 26, "plastic", "general_trash", "DIRECT", "NO", "Expanded polystyrene takeaway container; non-recyclable curbside", "HIGH"],
    ["Styrofoam piece", 195, 195, "plastic", "general_trash", "DIRECT", "NO", "Expanded polystyrene fragments; non-recyclable residual waste", "HIGH"],
    ["Plastic straw", 280, 280, "plastic", "general_trash", "DIRECT", "NO", "Single-use plastic straws are banned/residual waste under India PWM Rules", "HIGH"],
    ["Plastic utensils", 60, 60, "plastic", "general_trash", "DIRECT", "NO", "Single-use plastic cutlery is banned/residual waste under India PWM Rules", "HIGH"],
    ["Squeezable tube", 12, 12, "plastic", "general_trash", "DIRECT", "NO", "Multi-material composite laminate (toothpaste/cosmetics); non-recyclable", "HIGH"],
    ["Plastic glooves", 6, 6, "plastic", "general_trash", "DIRECT", "NO", "Single-use cleaning/medical gloves are sanitary waste; special care handling", "HIGH"],
    ["Disposable food container", 62, 62, "plastic", "general_trash", "CONDITIONAL", "YES", "Visibly food-soiled takeout containers require residual disposal", "MEDIUM"],
    # Metal
    ["Drink can", 370, 370, "metal", "metal", "DIRECT", "NO", "Aluminium beverage cans accepted in Blue Bin", "HIGH"],
    ["Food Can", 55, 55, "metal", "metal", "DIRECT", "NO", "Steel/tin food cans accepted in Blue Bin", "HIGH"],
    ["Metal bottle cap", 133, 133, "metal", "metal", "DIRECT", "NO", "Crown bottle caps accepted with scrap metal", "HIGH"],
    ["Metal lid", 19, 19, "metal", "metal", "DIRECT", "NO", "Jar metal lids accepted for recycling", "HIGH"],
    ["Pop tab", 161, 161, "metal", "metal", "DIRECT", "NO", "Aluminium can pull tabs accepted for recycling", "HIGH"],
    ["Aerosol", 17, 17, "metal", "metal", "DIRECT", "NO", "Empty aluminium/steel aerosol cans accepted for recycling", "HIGH"],
    ["Scrap metal", 39, 39, "metal", "metal", "DIRECT", "NO", "Small clean metal hardware and scrap", "HIGH"],
    ["Aluminium foil", 106, 106, "metal", "metal", "CONDITIONAL", "YES", "Clean foil is recyclable metal; food-caked foil requires residual disposal", "MEDIUM"],
    # Cardboard
    ["Corrugated carton", 114, 114, "cardboard", "cardboard", "DIRECT", "NO", "Clean corrugated shipping cartons accepted in Blue Bin", "HIGH"],
    ["Egg carton", 21, 21, "cardboard", "cardboard", "DIRECT", "NO", "Molded pulp paperboard egg boxes", "HIGH"],
    ["Toilet tube", 9, 9, "cardboard", "cardboard", "DIRECT", "NO", "Cardboard inner cores", "HIGH"],
    ["Other carton", 159, 159, "cardboard", "cardboard", "DIRECT", "NO", "Dry paperboard packaging cartons", "HIGH"],
    ["Drink carton", 73, 73, "cardboard", "cardboard", "DIRECT", "NO", "Aseptic beverage cartons (Tetra Pak) accepted in paperboard hydrapulping", "HIGH"],
    ["Pizza box", 5, 5, "cardboard", "general_trash", "CONDITIONAL", "YES", "Greasy food-soiled pizza boxes must go to general_trash per MoEFCC SWM rules", "HIGH"],
    ["Meal carton", 57, 57, "cardboard", "general_trash", "CONDITIONAL", "YES", "Food-soiled takeout cartons must go to general_trash if oil-stained", "MEDIUM"],
    # Glass
    ["Glass bottle", 181, 181, "glass", "glass", "DIRECT", "NO", "Container glass bottles accepted in glass stream", "HIGH"],
    ["Glass jar", 11, 11, "glass", "glass", "DIRECT", "NO", "Container glass jars accepted in glass stream", "HIGH"],
    ["Glass cup", 11, 11, "glass", "glass", "DIRECT", "NO", "Drinking glassware (handled in container glass)", "MEDIUM"],
    ["Broken glass", 193, 193, "glass", "glass", "DIRECT", "NO", "Broken container glass; requires protective paper wrapping guidance", "HIGH"],
    # Paper
    ["Normal paper", 136, 136, "paper", "paper", "DIRECT", "NO", "Clean office copy paper sheets accepted in Blue Bin", "HIGH"],
    ["Magazine paper", 14, 14, "paper", "paper", "DIRECT", "NO", "Glossy printed paper publications", "HIGH"],
    ["Paper bag", 51, 51, "paper", "paper", "DIRECT", "NO", "Clean kraft shopping bags", "HIGH"],
    ["Paper straw", 7, 7, "paper", "paper", "DIRECT", "NO", "Clean paper drinking straws", "HIGH"],
    ["Wrapping paper", 17, 17, "paper", "paper", "DIRECT", "NO", "Clean decorative gift wrapping paper", "HIGH"],
    ["Paper cup", 117, 117, "paper", "paper", "DIRECT", "NO", "Clean paper drinking cups", "MEDIUM"],
    ["Tissues", 75, 75, "paper", "general_trash", "DIRECT", "NO", "Used facial tissues/napkins are sanitary bio-waste; special care packaging", "HIGH"],
    # General Trash (Residual & Tanglers)
    ["Cigarette", 1128, 1128, "general_trash", "general_trash", "DIRECT", "NO", "Cellulose acetate filters and ash; non-recyclable residual litter", "HIGH"],
    ["Rope - strings", 51, 51, "general_trash", "general_trash", "DIRECT", "NO", "Synthetic cords and tanglers; residual landfill waste", "HIGH"],
    ["Aluminium blister pack", 12, 12, "general_trash", "general_trash", "DIRECT", "NO", "Foil/PVC pharmaceutical blister pack; non-recyclable composite", "HIGH"],
    ["Carded blister pack", 2, 2, "null", "general_trash", "DIRECT", "NO", "Paper/plastic composite blister packaging; non-recyclable", "HIGH"],
    ["Unlabeled litter", 903, 903, "general_trash", "general_trash", "MANUAL_REVIEW", "YES", "Ambiguous small litter fragments; requires verification against noise", "LOW"],
    # Organic (Wet Waste)
    ["Food waste", 13, 13, "null", "organic", "DIRECT", "NO", "Biodegradable food scraps; routed to Green Bin for composting", "HIGH"],
    ["BIODEGRADABLE (GWD)", 10464, 45395, "null", "organic", "DIRECT", "NO", "Fruit peels, vegetable trimmings, food scraps from General Waste Dataset", "HIGH"],
    # Excluded
    ["Battery", 4, 4, "null", "null", "EXCLUDED", "NO", "Hazardous e-waste; domestic hazardous separate drop-off", "HIGH"],
    ["Shoe", 11, 11, "null", "null", "EXCLUDED", "NO", "Footwear; excluded from municipal packaging/sorting benchmark", "HIGH"]
]

mapping_csv_path = OUTPUT_DIR / "mapping_review.csv"
with open(mapping_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "source_category", "source_image_count", "source_instance_count",
        "current_class", "proposed_v5_class", "mapping_type",
        "review_required", "reason", "confidence"
    ])
    for row in MAPPING_REVIEW_ROWS:
        writer.writerow(row)

print(f"Saved {mapping_csv_path}")

# 2. Build manual_review_queue.csv
# Columns: image_path, annotation_id, source_category, proposed_class, review_reason, review_status, final_decision, reviewer_notes

manual_queue_path = OUTPUT_DIR / "manual_review_queue.csv"
with open(manual_queue_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path", "annotation_id", "source_category", "proposed_class",
        "review_reason", "review_status", "final_decision", "reviewer_notes"
    ])
    # Review category criteria definitions:
    review_criteria = [
        ["CRITERIA_PIZZA_BOX", "N/A", "Pizza box", "general_trash", "Grease staining assessment: if greasy -> general_trash; if clean dry lid -> cardboard", "APPROVED_RULE", "general_trash", "Under MoEFCC/CPCB SWM 2026, food-contact pizza boxes are contaminated; mapped to general_trash."],
        ["CRITERIA_TISSUES", "N/A", "Tissues", "general_trash", "Sanitary classification: used facial tissues/napkins are non-recyclable bio-waste", "APPROVED_RULE", "general_trash", "Under MoEFCC/CPCB SWM 2026, tissues are Sanitary Waste requiring special care / general disposal."],
        ["CRITERIA_FOAM_CONTAINERS", "N/A", "Foam food container / cup", "general_trash", "Expanded polystyrene is non-recyclable curbside in Indian municipalities", "APPROVED_RULE", "general_trash", "Mapped to general_trash per MoEFCC PWM residual waste standard."],
        ["CRITERIA_SNACK_PACKETS", "N/A", "Crisp packet", "general_trash", "Metallized BOPP multilayer composite film cannot be recycled with rigid plastics", "APPROVED_RULE", "general_trash", "Mapped to general_trash per MoEFCC PWM residual waste standard."],
        ["CRITERIA_MEAL_CARTON", "N/A", "Meal carton", "general_trash", "Food-soiled takeout paperboard container", "APPROVED_RULE", "general_trash", "Takeout cartons with food grease routed to general_trash per MoEFCC SWM Rules."],
        ["CRITERIA_DISPOSABLE_FOOD_CONTAINER", "N/A", "Disposable food container", "general_trash", "Plastic clamshell with food grease residues", "APPROVED_RULE", "general_trash", "Food-contact takeaway containers routed to general_trash per MoEFCC SWM Rules."],
        ["CRITERIA_UNLABELED_LITTER", "N/A", "Unlabeled litter", "general_trash", "Sub-10px outdoor debris fragments; filter degenerate boxes (<4px) before V5 packaging", "APPROVED_RULE", "general_trash", "Retain valid litter boxes above 4x4 px; drop degenerate coordinates."]
    ]
    for row in review_criteria:
        writer.writerow(row)

print(f"Saved {manual_queue_path}")

