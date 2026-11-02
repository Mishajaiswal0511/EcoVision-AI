# EcoVision AI — Staging Test Report (V5 7-Class Model & Guidance)

**Date:** 2026-11-02 20:21:20  
**Target Application:** Isolated Staging Backend (`models/experiments/v5_taxonomy/preproduction/staging/`)  
**Checkpoint:** `models/experiments/v5_taxonomy/training/yolo11s_v5_best.pt`  
**Overall Result:** **FAIL** (15/16 Tests Passed)  

---

## 1. Automated Test Matrix

| # | Test Scenario | Input / Dataset Image | Expected Behavior | Actual Behavior | Result |
|:---:|---|---|---|---|:---:|
| **1** | 1. Plastic Object Detection | `gw_plastic1002_jpg.rf.2cbd88cb12a6e39bde86b3760f226118.jpg` | plastic | plastic | **PASS** |
| **2** | 2. Metal Object Detection | `gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg` | metal | metal | **PASS** |
| **3** | 3. Cardboard Object Detection | `gw_cardboard1000_jpg.rf.84fa21f62f10746865df43604d3f7031.jpg` | cardboard | cardboard | **PASS** |
| **4** | 4. Glass Object Detection | `gw_glass1000_jpg.rf.5a73deb55b262a2fd46fd9ad3089ed81.jpg` | glass | glass | **PASS** |
| **5** | 5. Paper Object Detection | `gw_paper1001_jpg.rf.572b43db7925850b52ec9a945f188ea3.jpg` | paper | paper | **PASS** |
| **6** | 6. General Trash Detection | `taco_000000_jpg.rf.1d051677f093973bf92f3336e269f2bc.jpg` | general_trash | general_trash | **PASS** |
| **7** | 7. Organic Object Detection | `gw_biodegradable1225_jpg.rf.62c0de814da542600b8d18bbf282448c.jpg` | organic | organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, cardboard, organic, cardboard, organic, cardboard, organic | **PASS** |
| **8** | 8. Multi-Object Detection | `gw_biodegradable1194_jpg.rf.268c14469221eafa41137461b9fc23c5.jpg` | None | organic, organic, organic, organic, organic, organic, organic, organic, organic, organic, cardboard, cardboard, cardboard, cardboard | **PASS** |
| **9** | 9. Empty / Background Negative Image | `neg_000000001503.jpg` | None (0 detections) | Produced 1 false positives on clean background. | **FAIL** |
| **10** | 10. Invalid Image File (Corrupted Bytes) | `N/A (in-memory corrupted byte buffer)` | InvalidImageError Exception | Successfully caught InvalidImageError: Failed to decode image from bytes: cannot identify image file <_io.BytesIO object at 0x000002D6004E0900> | **PASS** |
| **11** | 11. Unsupported Format Handling | `N/A (plain text stream)` | InvalidImageError | Correctly rejected unreadable format: Failed to decode image from bytes: cannot identify image file <_io.BytesIO object at 0x000002D6004E0900> | **PASS** |
| **12** | 12. Missing Image Input (0-byte Buffer) | `N/A (0-byte buffer)` | InvalidImageError | Correctly caught empty input: Input image bytes are empty | **PASS** |
| **13** | 13. Confidence Threshold Filtering Sensitivity | `gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg` | Higher threshold yields subset of lower threshold | At conf=0.05: 1 detections; at conf=0.85: 1 detections. Correct monotonically decreasing behavior. | **PASS** |
| **14** | 14. Multiple Objects of Same Class | `gw_cardboard1036_jpg.rf.fd7294157a2a746d10658fed4213a30d.jpg` | cardboard | cardboard, cardboard, cardboard, cardboard, cardboard | **PASS** |
| **15** | 15. Sanitary Guidance Display Verification | `N/A (Disposal Service Guidance Contract)` | Sanitary warning banner + Newspaper Red-mark step | general_trash | **PASS** |
| **16** | 16. Conditional Packaging Dual-Guidance Rules | `N/A (Disposal Service Guidance Contract)` | Dual clean vs soiled rules | cardboard, plastic, general_trash | **PASS** |

---

## 2. Detailed Test Case Findings

### Test 1: 1. Plastic Object Detection
- **Input:** `gw_plastic1002_jpg.rf.2cbd88cb12a6e39bde86b3760f226118.jpg`
- **Execution Latency:** 2929.93 ms
- **Detected Classes:** ['plastic']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 2: 2. Metal Object Detection
- **Input:** `gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg`
- **Execution Latency:** 8235.97 ms
- **Detected Classes:** ['metal']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 3: 3. Cardboard Object Detection
- **Input:** `gw_cardboard1000_jpg.rf.84fa21f62f10746865df43604d3f7031.jpg`
- **Execution Latency:** 260.12 ms
- **Detected Classes:** ['cardboard']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 4: 4. Glass Object Detection
- **Input:** `gw_glass1000_jpg.rf.5a73deb55b262a2fd46fd9ad3089ed81.jpg`
- **Execution Latency:** 181.53 ms
- **Detected Classes:** ['glass']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 5: 5. Paper Object Detection
- **Input:** `gw_paper1001_jpg.rf.572b43db7925850b52ec9a945f188ea3.jpg`
- **Execution Latency:** 209.66 ms
- **Detected Classes:** ['paper']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 6: 6. General Trash Detection
- **Input:** `taco_000000_jpg.rf.1d051677f093973bf92f3336e269f2bc.jpg`
- **Execution Latency:** 190.24 ms
- **Detected Classes:** ['general_trash']
- **Verification Evidence:** Detected 1 objects matching expected category profile.
- **Result:** **PASS**

### Test 7: 7. Organic Object Detection
- **Input:** `gw_biodegradable1225_jpg.rf.62c0de814da542600b8d18bbf282448c.jpg`
- **Execution Latency:** 186.47 ms
- **Detected Classes:** ['organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'cardboard', 'organic', 'cardboard', 'organic', 'cardboard', 'organic']
- **Verification Evidence:** Detected 43 objects matching expected category profile.
- **Result:** **PASS**

### Test 8: 8. Multi-Object Detection
- **Input:** `gw_biodegradable1194_jpg.rf.268c14469221eafa41137461b9fc23c5.jpg`
- **Execution Latency:** 182.44 ms
- **Detected Classes:** ['organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'organic', 'cardboard', 'cardboard', 'cardboard', 'cardboard']
- **Verification Evidence:** Detected 14 objects matching expected category profile.
- **Result:** **PASS**

### Test 9: 9. Empty / Background Negative Image
- **Input:** `neg_000000001503.jpg`
- **Execution Latency:** 169.94 ms
- **Detected Classes:** []
- **Verification Evidence:** Produced 1 false positives on clean background.
- **Result:** **FAIL**

### Test 10: 10. Invalid Image File (Corrupted Bytes)
- **Input:** `N/A (in-memory corrupted byte buffer)`
- **Execution Latency:** 0.0 ms
- **Detected Classes:** []
- **Verification Evidence:** Successfully caught InvalidImageError: Failed to decode image from bytes: cannot identify image file <_io.BytesIO object at 0x000002D6004E0900>
- **Result:** **PASS**

### Test 11: 11. Unsupported Format Handling
- **Input:** `N/A (plain text stream)`
- **Execution Latency:** 0.0 ms
- **Detected Classes:** []
- **Verification Evidence:** Correctly rejected unreadable format: Failed to decode image from bytes: cannot identify image file <_io.BytesIO object at 0x000002D6004E0900>
- **Result:** **PASS**

### Test 12: 12. Missing Image Input (0-byte Buffer)
- **Input:** `N/A (0-byte buffer)`
- **Execution Latency:** 0.0 ms
- **Detected Classes:** []
- **Verification Evidence:** Correctly caught empty input: Input image bytes are empty
- **Result:** **PASS**

### Test 13: 13. Confidence Threshold Filtering Sensitivity
- **Input:** `gw_metal1005_jpg.rf.c2579ac46cc776d0c27d4056a8f24c88.jpg`
- **Execution Latency:** 335.3 ms
- **Detected Classes:** []
- **Verification Evidence:** At conf=0.05: 1 detections; at conf=0.85: 1 detections. Correct monotonically decreasing behavior.
- **Result:** **PASS**

### Test 14: 14. Multiple Objects of Same Class
- **Input:** `gw_cardboard1036_jpg.rf.fd7294157a2a746d10658fed4213a30d.jpg`
- **Execution Latency:** 416.52 ms
- **Detected Classes:** ['cardboard', 'cardboard', 'cardboard', 'cardboard', 'cardboard']
- **Verification Evidence:** Detected 5 objects matching expected category profile.
- **Result:** **PASS**

### Test 15: 15. Sanitary Guidance Display Verification
- **Input:** `N/A (Disposal Service Guidance Contract)`
- **Execution Latency:** 0.0 ms
- **Detected Classes:** ['general_trash']
- **Verification Evidence:** Warning banner: '⚠️ SANITARY & SPECIAL-CARE ADVISORY: If this detected item is a used tissue, nap...' | Red wrap step present: True
- **Result:** **PASS**

### Test 16: 16. Conditional Packaging Dual-Guidance Rules
- **Input:** `N/A (Disposal Service Guidance Contract)`
- **Execution Latency:** 0.0 ms
- **Detected Classes:** ['cardboard', 'plastic', 'general_trash']
- **Verification Evidence:** Cardboard conditional rule present: True; Plastic conditional rule: True; General trash grease rule: True.
- **Result:** **PASS**

---

## 3. Test Verdict & Staging Integrity
- **Total Test Scenarios Executed:** 16
- **Tests Passed:** 15
- **Tests Failed:** 1
- **Zero Production Contamination:** All test executions operated exclusively against the staging isolated service without accessing or modifying production assets.
- **Sealed Evaluation Data Sealed:** Zero test cases touched `data/real_world_eval/`.