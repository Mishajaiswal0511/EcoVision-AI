# EcoVision AI — Governing Municipal Solid Waste Management Policy Specification

**Document:** `docs/municipal_policy/policy_source.md`  
**Governing Authority:** Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India  
**Co-Regulators:** Central Pollution Control Board (CPCB) & Department of Local Government, Government of Punjab  
**Statutory Framework:** Solid Waste Management Rules (MoEFCC / CPCB) with Punjab Municipal SWM Implementation Guidelines  
**Adopted Version:** SWM National Regulatory Standard (2026 Revision)  
**Effective Date:** 2026-10-31  

---

## 1. Statutory Waste Segregation Architecture

Under the governing MoEFCC / CPCB framework, solid waste must be segregated at source into distinct municipal streams:

| Stream Name | Visual Bin Color Standard | Material Scope | EcoVision AI V5 Target Class Mapping |
|---|:---:|---|:---:|
| **Wet Waste (Biodegradable)** | **Green Bin** | Cooked and uncooked food scraps, vegetable and fruit peels, tea leaves, eggshells, garden/leaf waste. | **Class 6: `organic`** |
| **Dry Recyclable Waste** | **Blue Bin** | Clean, non-contaminated plastic packaging, metal cans, cardboard boxes, paper sheets, and container glass. | **Class 0: `plastic`**<br>**Class 1: `metal`**<br>**Class 2: `cardboard`**<br>**Class 3: `glass`**<br>**Class 4: `paper`** |
| **Residual / Inert Waste (General Trash)** | **Black / Grey Bin** | Non-recyclable composite packaging, multilayer snack wrappers, expanded polystyrene (Styrofoam), non-recyclable sweeping inerts, cigarette butts. | **Class 5: `general_trash`** |
| **Sanitary Waste** | **Dedicated Pouch / Distinct Red Marking** | Soiled diapers, sanitary pads, used tissues, bandages, single-use medical/cleaning gloves. **Never mixed into bulk dry recycling.** | **Special Handling Stream** (Routed to incinerator / deep burial per municipal rules) |
| **Special Care / Domestic Hazardous Waste** | **Specialized Collection / Safe Depository** | Batteries, e-waste, aerosol cans with residual gas, broken glass, chemical containers. | **Special Handling Stream** (Specialized drop-off / safety wrapping) |

---

## 2. Stream-by-Stream Material Rules

### A. Wet Waste (Organic) — Class 6
- **Accepted:** Kitchen food waste, raw fruit & vegetable peels, spent tea & coffee grounds, floral and garden trimmings, eggshells.
- **Prohibited:** Non-biodegradable plastic packaging, plastic shopping bags, foil wrapping, sanitary pads, diapers.
- **Handling:** Collected daily for decentralized composting or centralized biomethanation / anaerobic digestion. Never dispose in conventional plastic carrier bags unless certified compostable (IS/ISO 17088 compliant).

### B. Dry Recyclables — Classes 0 to 4
- **Cleanliness Standard:** Dry waste must be rinsed and emptied of food residues.
- **Class 0 (`plastic`):** Clean PET beverage bottles, HDPE containers, clean PP tubs, clean polyethylene film. *Excludes:* Multilayer metallized snack packets and expanded polystyrene, which go to Class 5 (`general_trash`).
- **Class 1 (`metal`):** Clean aluminum cans, steel/tin food cans, jar lids, crown caps, pop tabs, clean aluminum foil. *Excludes:* Dry cell and lithium batteries (Domestic Hazardous).
- **Class 2 (`cardboard`):** Clean corrugated shipping cartons, cereal boxes, molded pulp egg crates. *Excludes:* Greasy food-encrusted pizza boxes or oil-soaked boxes, which go to Class 5 (`general_trash`).
- **Class 3 (`glass`):** Transparent and colored container glass (bottles, food jars). *Safety Rule:* Broken glass must be wrapped in newspaper / designated thick paper before placement to prevent cut injuries to sanitation workers. Window glass and ceramics are excluded from container recycling.
- **Class 4 (`paper`):** Dry office paper, newspapers, magazines, clean paper bags, notebooks. *Excludes:* Used facial tissues and paper napkins, which are sanitary waste.

### C. General Trash (Residual Waste) — Class 5
- **Accepted:** Multilayer composite snack/crisp packets (metallized BOPP/PET film), expanded polystyrene (Styrofoam clamshells & cups), food-contaminated paperboard, cigarette butts, non-recyclable composite packaging, tangled ropes/cords.
- **Routing:** Destined for Refuse Derived Fuel (RDF) processing, cement co-processing, waste-to-energy facilities, or engineered sanitary landfills.

### D. Sanitary & Special-Care Waste Handling (Statutory Requirement)
- In strict adherence to MoEFCC and Punjab municipal guidelines, **Sanitary Waste is NOT classified as ordinary General Trash**.
- The EcoVision AI guidance system must explicitly mandate:
  1. **Sanitary Waste (Tissues, Diapers, Gloves):** Must be securely wrapped in newspaper or designated paper bags, marked with a distinctive red mark, and handed over separately to municipal sanitation workers.
  2. **Domestic Hazardous (Batteries):** Must never enter regular bins. Route to designated e-waste drop-off kiosks or municipal hazardous waste drives.
  3. **Broken Glass:** Wrap securely in thick newspaper and label clearly before placing in glass bank or handing over.

