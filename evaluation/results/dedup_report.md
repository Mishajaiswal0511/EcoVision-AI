# EcoVision AI — Deduplication Report

> Generated : 2026-10-29T00:15:23.227610
> Status    : **🔴 ACTION NEEDED**
> pHash threshold : 10 (Hamming distance)

---

## Summary

| Metric | Value |
|---|---|
| Images scanned | 2550 |
| Hash errors | 0 |
| **Exact duplicate groups** | **0** |
| Exact redundant images | 0 |
| **Near-duplicate groups** | **120** |
| Near-duplicate redundant images | 128 |
| **Cross-split contamination groups** | **5** |

> [!CAUTION]
> Cross-split contamination means the same (or near-identical) image
> appears in both the training set and the test/validation set.
> This inflates evaluation scores and must be resolved before training.

## Deduplication Plan (Top 20 Groups)

_Full plan in `dedup_plan.json`. Only top 20 shown here._

### Group 1
- **Keeper** (`train`): `000000_jpg.rf.1d051677f093973bf92f3336e269f2bc.jpg`
- Candidate for removal (`train`): `000000_jpg.rf.283942c7cf645372b84351e6be9dffe3.jpg`

### Group 2
- **Keeper** (`train`): `000001_jpg.rf.0763006f9a7cfcb795604842741f1992.jpg`
- Candidate for removal (`train`): `000001_jpg.rf.7d2ec55b6a16af5d8fb49a71fffbea42.jpg`

### Group 3
- **Keeper** (`train`): `000002_JPG.rf.6d681994073962f4e3789a0e160079c7.jpg`
- Candidate for removal (`train`): `000002_JPG.rf.93ffc2a8add35eea938c333324a60c8b.jpg`

### Group 4
- **Keeper** (`train`): `000002_JPG.rf.a0f798a799f483b602a782a3b827736c.jpg`
- Candidate for removal (`train`): `000002_JPG.rf.afe76be02f910cca8ea7712b6b736541.jpg`

### Group 5
- **Keeper** (`train`): `000003_jpg.rf.1b2815de8f8440eb0f4942707feea2bf.jpg`
- Candidate for removal (`train`): `000003_jpg.rf.4b1218eb06c78ca1d26dde5021fd3336.jpg`

### Group 6
- **Keeper** (`train`): `000003_jpg.rf.a6a692e893bf1f519b796ac14dac032f.jpg`
- Candidate for removal (`train`): `000003_jpg.rf.b3f3ec9232a4068106068d4acace067f.jpg`

### Group 7
- **Keeper** (`train`): `000004_jpg.rf.57007580e4a8fe98057034c2db94c2dd.jpg`
- Candidate for removal (`train`): `000004_jpg.rf.c116dab73dad8217dfbc199ceb6f4401.jpg`

### Group 8
- **Keeper** (`train`): `000005_jpg.rf.02f41188916f6890b83b03a909df9e03.jpg`
- Candidate for removal (`train`): `000005_jpg.rf.6df61b8f6fb6d9dc9fae456207ed4a28.jpg`

### Group 9 🔴 CROSS-SPLIT
- **Keeper** (`train`): `000005_jpg.rf.41eb3e5de359efd2741dfcdd4cb901dd.jpg`
- Candidate for removal (`val`): `000000_JPG.rf.04216bd9f433d6eb121f48686d6cee32.jpg`

### Group 10
- **Keeper** (`train`): `000005_JPG.rf.a1773593f3f9bcf779b5623ba689bf34.jpg`
- Candidate for removal (`train`): `000005_JPG.rf.dcf50c8a5a4085afda11d0aefb23bc45.jpg`

### Group 11 🔴 CROSS-SPLIT
- **Keeper** (`train`): `000005_JPG.rf.c346472521451f78a3a2a8826b0d838c.jpg`
- Candidate for removal (`test`): `000064_jpg.rf.17f1f7891a536096acb7ff3a8005fa45.jpg`

### Group 12
- **Keeper** (`train`): `000006_jpg.rf.69f46ab8b76cddbb0dd077e5eb5f1b74.jpg`
- Candidate for removal (`train`): `000006_jpg.rf.d197c178aca9e9dd148edd543243a9d3.jpg`

### Group 13
- **Keeper** (`train`): `000007_jpg.rf.039b924eae5a35eebf95632c27ed084c.jpg`
- Candidate for removal (`train`): `000007_jpg.rf.9705367b77d2db78e85f2fcdeea52cfa.jpg`

### Group 14
- **Keeper** (`train`): `000008_jpg.rf.032a352f582d6f2a063b5719a3269d7c.jpg`
- Candidate for removal (`train`): `000008_jpg.rf.04ac345bb23d809bab418b82197f1987.jpg`

### Group 15
- **Keeper** (`train`): `000010_JPG.rf.0baf7a8b9f5fcb47a9b732d168ae3fbc.jpg`
- Candidate for removal (`train`): `000010_JPG.rf.42093adbee64125ac39c40ff631c8b96.jpg`

### Group 16
- **Keeper** (`train`): `000010_jpg.rf.0f1d7f30f1e1bca5b341350fba9b01f9.jpg`
- Candidate for removal (`train`): `000010_jpg.rf.ef2f7d40eb71079f1369153ed644d335.jpg`

### Group 17
- **Keeper** (`train`): `000010_JPG.rf.5489c4e2abdcdf2c23a1f51c2ddf469f.jpg`
- Candidate for removal (`train`): `000010_JPG.rf.a108680721e1cda5886a29bf9e10754c.jpg`

### Group 18
- **Keeper** (`train`): `000011_JPG.rf.9bd697e28f5f01432c4928b83428be74.jpg`
- Candidate for removal (`train`): `000011_JPG.rf.a025603fc34f07d4ae017ea74c064a5f.jpg`

### Group 19
- **Keeper** (`train`): `000012_JPG.rf.7a7514204ecee7c892aaee94909c2ed8.jpg`
- Candidate for removal (`train`): `000012_JPG.rf.b6c0c3d757787b9468d4c862e8a26058.jpg`

### Group 20
- **Keeper** (`train`): `000015_jpg.rf.3d92172df21dcd2e06d6170545d9e70d.jpg`
- Candidate for removal (`train`): `000015_jpg.rf.4cede84492f1ca011ab2f175065e48d3.jpg`

---

> **Important:** No files have been deleted. This report is for review only.
> After manual verification, remove duplicates using your preferred method.
> Prioritise removing cross-split duplicates first.