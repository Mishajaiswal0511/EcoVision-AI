# EcoVision AI — YOLO Detection Annotation Quality Report

**Generated:** 2026-10-29 18:33:16
**Classes (6):** plastic, metal, cardboard, glass, paper, general_trash
**Coordinate Boundary Tolerance:** `0.0001`

---

## Summary by Split

| Split | Images | Labels | Annotations Scanned | Valid Annotations | Field Count Errors | Invalid Class IDs | Out-of-Bounds | Degenerate (w/h<=0) | Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 1790 | 1790 | 5939 | 5939 | 0 | 0 | 0 | 0 | 0 |
| `val` | 379 | 379 | 1077 | 1077 | 0 | 0 | 0 | 0 | 0 |
| `test` | 374 | 374 | 1106 | 1106 | 0 | 0 | 0 | 0 | 0 |

---

## Detailed Issues by Split

### Split: `train`
#### Tiny Box Warnings (366)
- `{"file": "000000_jpg.rf.01d9b88b126b535f2fbcfcbed0985b53.txt", "line": 5, "area": 8.857e-05}`
- `{"file": "000000_jpg.rf.5f4f3ad526c184c535ea751e42877e2b.txt", "line": 5, "area": 8.857e-05}`
- `{"file": "000001_jpg.rf.b8baf2e51b9c3a2eed6651e1f77d86bb.txt", "line": 10, "area": 5.263e-05}`
- `{"file": "000001_jpg.rf.ba62e8d0aaacb7afc3ca47696a5b9b5a.txt", "line": 10, "area": 5.263e-05}`
- `{"file": "000002_JPG.rf.610e0ec78d4d5adc82178d691606dc70.txt", "line": 5, "area": 5.607e-05}`
- *... and 361 more*

### Split: `val`
#### Tiny Box Warnings (47)
- `{"file": "000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 5, "area": 5.607e-05}`
- `{"file": "000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 6, "area": 9.01e-05}`
- `{"file": "000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 8, "area": 5.206e-05}`
- `{"file": "000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 13, "area": 4.255e-05}`
- `{"file": "000002_JPG.rf.e0a28580310b55e0d065bf9eb5fc5442.txt", "line": 14, "area": 9.462e-05}`
- *... and 42 more*

### Split: `test`
#### Tiny Box Warnings (39)
- `{"file": "000014_jpg.rf.724d956c14d3471e9ee7a64050631fea.txt", "line": 19, "area": 7.009e-05}`
- `{"file": "000015_jpg.rf.e451ae5896a23d7df68bf12be160dca8.txt", "line": 6, "area": 4.055e-05}`
- `{"file": "000021_jpg.rf.cc2b702e2b680b06fc333a3dd3f0ee56.txt", "line": 4, "area": 6.693e-05}`
- `{"file": "000023_jpg.rf.3b7271734f1fbdd046b193e9824b270b.txt", "line": 11, "area": 9.699e-05}`
- `{"file": "000023_jpg.rf.3b7271734f1fbdd046b193e9824b270b.txt", "line": 12, "area": 9.912e-05}`
- *... and 34 more*
