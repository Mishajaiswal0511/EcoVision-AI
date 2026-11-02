# EcoVision AI — YOLO Detection Annotation Quality Report

**Generated:** 2026-10-29 20:38:27
**Classes (6):** plastic, metal, cardboard, glass, paper, general_trash
**Coordinate Boundary Tolerance:** `0.0001`

---

## Summary by Split

| Split | Images | Labels | Annotations Scanned | Valid Annotations | Field Count Errors | Invalid Class IDs | Out-of-Bounds | Degenerate (w/h<=0) | Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 8242 | 8242 | 25774 | 25774 | 0 | 0 | 0 | 0 | 0 |
| `val` | 3007 | 3007 | 9781 | 9781 | 0 | 0 | 0 | 0 | 0 |
| `test` | 374 | 374 | 1106 | 1106 | 0 | 0 | 0 | 0 | 0 |

---

## Detailed Issues by Split

### Split: `train`
#### Tiny Box Warnings (366)
- `{"file": "taco_000000_jpg.rf.01d9b88b126b535f2fbcfcbed0985b53.txt", "line": 5, "area": 8.857e-05}`
- `{"file": "taco_000000_jpg.rf.5f4f3ad526c184c535ea751e42877e2b.txt", "line": 5, "area": 8.857e-05}`
- `{"file": "taco_000001_jpg.rf.b8baf2e51b9c3a2eed6651e1f77d86bb.txt", "line": 10, "area": 5.263e-05}`
- `{"file": "taco_000001_jpg.rf.ba62e8d0aaacb7afc3ca47696a5b9b5a.txt", "line": 10, "area": 5.263e-05}`
- `{"file": "taco_000002_JPG.rf.610e0ec78d4d5adc82178d691606dc70.txt", "line": 5, "area": 5.607e-05}`
- *... and 361 more*

#### Large Box Warnings (732)
- `{"file": "gw_cardboard1001_jpg.rf.814010e3a552071cb545e72a9d5b65d2.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1004_jpg.rf.f0147c25a909adb18bd134e00c657b63.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard100_jpg.rf.abd605fcddf1d70e751b3a67c0f269c9.txt", "line": 3, "area": 0.9953}`
- `{"file": "gw_cardboard1010_jpg.rf.053ae7ab1547e1e1de2cd3b7c2df9b7a.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1013_jpg.rf.8942fd47c0c909258dd09e4b04c96a2f.txt", "line": 1, "area": 0.9953}`
- *... and 727 more*

### Split: `val`
#### Tiny Box Warnings (48)
- `{"file": "gw_biodegradable825_jpg.rf.85d9b80567aca67d708c485aba9d7838.txt", "line": 3, "area": 8.057e-05}`
- `{"file": "taco_000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 5, "area": 5.607e-05}`
- `{"file": "taco_000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 6, "area": 9.01e-05}`
- `{"file": "taco_000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 8, "area": 5.206e-05}`
- `{"file": "taco_000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 13, "area": 4.255e-05}`
- *... and 43 more*

#### Large Box Warnings (306)
- `{"file": "gw_cardboard1000_jpg.rf.84fa21f62f10746865df43604d3f7031.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1002_jpg.rf.45a4eb579deca0ff24a10a016f1be496.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1003_jpg.rf.b89069bc505d8d42075840ab65cd8842.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1007_jpg.rf.5244eed9599c7d96973198e6e4fe41f7.txt", "line": 1, "area": 0.9953}`
- `{"file": "gw_cardboard1008_jpg.rf.0522b28a619a500458bfa7848b742a59.txt", "line": 1, "area": 0.9953}`
- *... and 301 more*

### Split: `test`
#### Tiny Box Warnings (39)
- `{"file": "taco_000014_jpg.rf.724d956c14d3471e9ee7a64050631fea.txt", "line": 19, "area": 7.009e-05}`
- `{"file": "taco_000015_jpg.rf.e451ae5896a23d7df68bf12be160dca8.txt", "line": 6, "area": 4.055e-05}`
- `{"file": "taco_000021_jpg.rf.cc2b702e2b680b06fc333a3dd3f0ee56.txt", "line": 4, "area": 6.693e-05}`
- `{"file": "taco_000023_jpg.rf.3b7271734f1fbdd046b193e9824b270b.txt", "line": 11, "area": 9.699e-05}`
- `{"file": "taco_000023_jpg.rf.3b7271734f1fbdd046b193e9824b270b.txt", "line": 12, "area": 9.912e-05}`
- *... and 34 more*
