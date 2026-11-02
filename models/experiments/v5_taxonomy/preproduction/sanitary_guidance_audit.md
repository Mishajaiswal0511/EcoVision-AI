# EcoVision AI — Sanitary & Special-Care Guidance Audit

**Audit Date:** 2026-11-01  
**Scope:** V5 Staging Disposal Guidance Architecture for Sanitary and Special-Care Waste.  
**File Modified:** [`models/experiments/v5_taxonomy/application_staging/disposal_guide.json`](file:///d:/Projects/ml_project/472/models/experiments/v5_taxonomy/application_staging/disposal_guide.json)  
**Governing Authority:** Ministry of Environment, Forest and Climate Change (MoEFCC) / Central Pollution Control Board (CPCB) Solid Waste Management Rules 2026; Department of Local Government, Punjab.  
**Status:** **RESOLVED & STAGED**  
**Audit Verdict:** **PASS**

---

## 1. Problem Statement & Taxonomy Constraint

The EcoVision AI V5 object detector is a **7-class single-head detector** (`plastic`, `metal`, `cardboard`, `glass`, `paper`, `general_trash`, `organic`).
Due to the absence of a dedicated `sanitary_waste` bounding-box class in underlying public benchmark datasets (such as TACO and GWD), the following sanitary and personal care objects were categorized into Class 5 (`general_trash`):
* Used facial tissues (75 instances in TACO)
* Paper napkins
* Single-use cleaning / protective gloves (6 instances in TACO)
* Hygiene and first-aid dressings

### The Guidance Risk
Previously, when the model detected an object as `general_trash`, the application retrieved instructions directing disposal into the **Black or Grey Bin for Landfill / Refuse Derived Fuel (RDF)** with `safety_warning: null`.
Under Indian Municipal Solid Waste Management statutory regulations, **sanitary waste must NEVER be mixed into general waste bins or landfill streams** due to bio-hazard infection risks to sanitation workers.

---

## 2. Statutory Policy Source Evidence

Under `docs/municipal_policy/policy_rules.json` and `docs/municipal_policy/policy_source.md`:
1. **Rule Statutory Mandate:** Solid waste generators must segregate sanitary waste at source.
2. **Handling Protocol:** Used tissues, napkins, diapers, and sanitary pads must be wrapped securely in newspaper or suitable paper pouches.
3. **Identification:** The wrapped package must be marked with a prominent **RED cross or dot**.
4. **Collection:** Handed over separately to municipal sanitation workers at the doorstep.
5. **Treatment Route:** Destined exclusively for Common Bio-medical Waste Treatment Facilities (CBWTF) or high-temperature incineration.

---

## 3. Before vs. After Guidance Comparison

### Before (V5 Staging Initial Draft):
```json
"general_trash": {
  "material": "Residual Waste (Non-Recyclable)",
  "bin_color_india": "Black or Grey",
  "bin_label": "General Waste (Landfill / RDF)",
  "preparation_steps": [
    "Includes multi-layer snack/crisp packets (BOPP/metallized film)",
    "Includes expanded polystyrene (Styrofoam cups and food clamshells)",
    "Includes greasy pizza boxes, soiled food containers, and single-use cutlery",
    "Includes cigarette butts, ropes, and small non-recyclable debris",
    "Bag securely to prevent spillage and litter dispersal"
  ],
  "disposal_locations": ["Black/Grey bin (general waste)", "Municipal refuse collection point"],
  "safety_warning": null,
  "recyclable": false,
  "compositable": false,
  "hazardous": false
}
```

### After (V5 Staging Updated):
```json
"general_trash": {
  "material": "Residual Waste (Non-Recyclable)",
  "bin_color_india": "Black or Grey",
  "bin_label": "General Waste (Landfill / RDF)",
  "preparation_steps": [
    "Includes non-recyclable composite packaging: multi-layer snack/crisp packets (metallized BOPP film), expanded polystyrene (Styrofoam cups/clamshells), and plastic straws/cutlery",
    "Includes heavily food-contaminated packaging: greasy pizza boxes and food-soiled meal cartons where oil/grease cannot be cleaned off",
    "Includes inert sweepings, cigarette butts, tangled ropes/strings, and outdoor non-recyclable litter fragments",
    "For standard non-sanitary residual waste: bag securely and place into the Black or Grey Bin (Landfill / RDF)",
    "SANITARY & SPECIAL-CARE EXCEPTION: If the detected object is a used tissue, paper napkin, disposable glove, diaper, or personal hygiene item, DO NOT dispose loose in regular general waste. Wrap securely in newspaper or paper pouch, mark with a prominent RED dot or cross, and hand over separately to municipal collectors per MoEFCC/CPCB SWM 2026 rules"
  ],
  "disposal_locations": ["Black/Grey bin (general waste)", "Municipal refuse collection point"],
  "safety_warning": "⚠️ SANITARY & SPECIAL-CARE ADVISORY: If this detected item is a used tissue, napkin, disposable glove, or personal hygiene item, DO NOT throw loose into the general waste bin. Statutory municipal guidelines require sanitary items to be wrapped securely in newspaper or paper pouch, marked with a distinctive RED cross or dot, and handed over separately to sanitation staff for high-temperature incineration.",
  "recyclable": false,
  "compositable": false,
  "hazardous": false
}
```

---

## 4. Test Case Validation

| Test Case | Detected Class | Displayed Warning in Staging UI | Conformance with Policy |
|---|:---:|---|:---:|
| **Used Tissue / Facial Wipe** | `general_trash` | Shows prominent amber `⚠️ SANITARY & SPECIAL-CARE ADVISORY` banner instructing newspaper wrap + red mark. | **PASS** |
| **Protective Cleaning Glove** | `general_trash` | Shows prominent amber `⚠️ SANITARY & SPECIAL-CARE ADVISORY` banner instructing separate handover. | **PASS** |
| **Crisp / Chip Packet** | `general_trash` | Guidance directs to Black/Grey bin (RDF); warning clarifies sanitary exception applies only if item is hygiene-related. | **PASS** |
| **Expanded Polystyrene Foam** | `general_trash` | Directs to Black/Grey bin (RDF) as ordinary residual waste. | **PASS** |

---

## 5. Audit Verdict & Safeguards
- **Verdict:** **PASS**
- **Safeguard Confirmation:** The live application file (`docs/disposal_guide.json`) was **NOT modified**. Only the staging file [`models/experiments/v5_taxonomy/application_staging/disposal_guide.json`](file:///d:/Projects/ml_project/472/models/experiments/v5_taxonomy/application_staging/disposal_guide.json) was updated.

