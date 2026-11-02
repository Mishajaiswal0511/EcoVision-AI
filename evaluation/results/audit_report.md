# EcoVision AI — Dataset Audit Report

> Generated: 2026-10-29T00:13:28.412280
> Data dir : `D:\Projects\ml_project\472\data\raw\garbage_taco-official-all-class_v6`
> Classes  : `D:\Projects\ml_project\472\data\raw\garbage_taco-official-all-class_v6\data.yaml`

---

## Global Summary

| Item | Count |
|---|---|
| Total images | 2550 |
| Total annotations | 8152 |
| Corrupt images | 0 |
| Missing label files | 0 |
| Invalid annotation files | 0 |

## Class Names

| ID | Name |
|---|---|
| 0 | `Aerosol` |
| 1 | `Aluminium blister pack` |
| 2 | `Aluminium foil` |
| 3 | `Battery` |
| 4 | `Broken glass` |
| 5 | `Carded blister pack` |
| 6 | `Cigarette` |
| 7 | `Clear plastic bottle` |
| 8 | `Corrugated carton` |
| 9 | `Crisp packet` |
| 10 | `Disposable food container` |
| 11 | `Disposable plastic cup` |
| 12 | `Drink can` |
| 13 | `Drink carton` |
| 14 | `Egg carton` |
| 15 | `Foam cup` |
| 16 | `Foam food container` |
| 17 | `Food Can` |
| 18 | `Food waste` |
| 19 | `Garbage bag` |
| 20 | `Glass bottle` |
| 21 | `Glass cup` |
| 22 | `Glass jar` |
| 23 | `Magazine paper` |
| 24 | `Meal carton` |
| 25 | `Metal bottle cap` |
| 26 | `Metal lid` |
| 27 | `Normal paper` |
| 28 | `Other carton` |
| 29 | `Other plastic` |
| 30 | `Other plastic bottle` |
| 31 | `Other plastic container` |
| 32 | `Other plastic cup` |
| 33 | `Other plastic wrapper` |
| 34 | `Paper bag` |
| 35 | `Paper cup` |
| 36 | `Paper straw` |
| 37 | `Pizza box` |
| 38 | `Plastic bottle cap` |
| 39 | `Plastic film` |
| 40 | `Plastic glooves` |
| 41 | `Plastic lid` |
| 42 | `Plastic straw` |
| 43 | `Plastic utensils` |
| 44 | `Polypropylene bag` |
| 45 | `Pop tab` |
| 46 | `Rope - strings` |
| 47 | `Scrap metal` |
| 48 | `Shoe` |
| 49 | `Single-use carrier bag` |
| 50 | `Six pack rings` |
| 51 | `Spread tub` |
| 52 | `Squeezable tube` |
| 53 | `Styrofoam piece` |
| 54 | `Tissues` |
| 55 | `Toilet tube` |
| 56 | `Tupperware` |
| 57 | `Unlabeled litter` |
| 58 | `Wrapping paper` |

---

## Split: `train`

| Metric | Value |
|---|---|
| Images | 2100 |
| Annotations | 6654 |
| Multi-object images | 1232 |
| Background images (no annotation) | 0 |
| Corrupt images | 0 |
| Missing label files | 0 |
| Invalid annotation files | 0 |

### Class Instance Distribution

| Class | Instances | Images containing class |
|---|---|---|
| `Aerosol` | 14 | 14 |
| `Aluminium blister pack` | 12 | 12 |
| `Aluminium foil` | 84 | 56 |
| `Battery` | 4 | 4 |
| `Broken glass` | 110 | 22 |
| `Carded blister pack` | 2 | 2 |
| `Cigarette` | 922 | 296 |
| `Clear plastic bottle` | 396 | 304 |
| `Corrugated carton` | 98 | 60 |
| `Crisp packet` | 54 | 50 |
| `Disposable food container` | 48 | 46 |
| `Disposable plastic cup` | 144 | 130 |
| `Drink can` | 278 | 208 |
| `Drink carton` | 56 | 54 |
| `Egg carton` | 20 | 16 |
| `Foam cup` | 20 | 16 |
| `Foam food container` | 22 | 18 |
| `Food Can` | 42 | 24 |
| `Food waste` | 10 | 8 |
| `Garbage bag` | 42 | 30 |
| `Glass bottle` | 152 | 108 |
| `Glass cup` | 10 | 6 |
| `Glass jar` | 10 | 6 |
| `Magazine paper` | 4 | 2 |
| `Meal carton` | 54 | 48 |
| `Metal bottle cap` | 106 | 78 |
| `Metal lid` | 18 | 10 |
| `Normal paper` | 108 | 90 |
| `Other carton` | 132 | 116 |
| `Other plastic` | 356 | 246 |
| `Other plastic bottle` | 78 | 70 |
| `Other plastic container` | 6 | 6 |
| `Other plastic cup` | 4 | 4 |
| `Other plastic wrapper` | 364 | 236 |
| `Paper bag` | 48 | 40 |
| `Paper cup` | 100 | 92 |
| `Paper straw` | 6 | 6 |
| `Pizza box` | 4 | 4 |
| `Plastic bottle cap` | 278 | 246 |
| `Plastic film` | 670 | 448 |
| `Plastic glooves` | 4 | 4 |
| `Plastic lid` | 122 | 108 |
| `Plastic straw` | 246 | 156 |
| `Plastic utensils` | 46 | 36 |
| `Polypropylene bag` | 6 | 6 |
| `Pop tab` | 106 | 86 |
| `Rope - strings` | 44 | 42 |
| `Scrap metal` | 38 | 22 |
| `Shoe` | 8 | 8 |
| `Single-use carrier bag` | 86 | 66 |
| `Six pack rings` | 20 | 6 |
| `Spread tub` | 14 | 12 |
| `Squeezable tube` | 10 | 8 |
| `Styrofoam piece` | 164 | 118 |
| `Tissues` | 66 | 56 |
| `Toilet tube` | 8 | 4 |
| `Tupperware` | 6 | 6 |
| `Unlabeled litter` | 764 | 374 |
| `Wrapping paper` | 10 | 10 |

## Split: `val`

| Metric | Value |
|---|---|
| Images | 300 |
| Annotations | 1029 |
| Multi-object images | 183 |
| Background images (no annotation) | 0 |
| Corrupt images | 0 |
| Missing label files | 0 |
| Invalid annotation files | 0 |

### Class Instance Distribution

| Class | Instances | Images containing class |
|---|---|---|
| `Aerosol` | 3 | 3 |
| `Aluminium blister pack` | 0 | 0 |
| `Aluminium foil` | 14 | 12 |
| `Battery` | 0 | 0 |
| `Broken glass` | 83 | 4 |
| `Carded blister pack` | 0 | 0 |
| `Cigarette` | 126 | 45 |
| `Clear plastic bottle` | 59 | 49 |
| `Corrugated carton` | 11 | 7 |
| `Crisp packet` | 11 | 9 |
| `Disposable food container` | 12 | 12 |
| `Disposable plastic cup` | 24 | 18 |
| `Drink can` | 43 | 25 |
| `Drink carton` | 9 | 8 |
| `Egg carton` | 1 | 1 |
| `Foam cup` | 2 | 2 |
| `Foam food container` | 3 | 3 |
| `Food Can` | 10 | 6 |
| `Food waste` | 3 | 3 |
| `Garbage bag` | 5 | 3 |
| `Glass bottle` | 22 | 14 |
| `Glass cup` | 1 | 1 |
| `Glass jar` | 1 | 1 |
| `Magazine paper` | 9 | 3 |
| `Meal carton` | 2 | 2 |
| `Metal bottle cap` | 13 | 10 |
| `Metal lid` | 1 | 1 |
| `Normal paper` | 20 | 17 |
| `Other carton` | 14 | 12 |
| `Other plastic` | 71 | 30 |
| `Other plastic bottle` | 9 | 8 |
| `Other plastic container` | 3 | 3 |
| `Other plastic cup` | 0 | 0 |
| `Other plastic wrapper` | 61 | 52 |
| `Paper bag` | 1 | 1 |
| `Paper cup` | 11 | 11 |
| `Paper straw` | 1 | 1 |
| `Pizza box` | 1 | 1 |
| `Plastic bottle cap` | 48 | 40 |
| `Plastic film` | 87 | 62 |
| `Plastic glooves` | 2 | 2 |
| `Plastic lid` | 12 | 11 |
| `Plastic straw` | 26 | 25 |
| `Plastic utensils` | 11 | 8 |
| `Polypropylene bag` | 0 | 0 |
| `Pop tab` | 27 | 14 |
| `Rope - strings` | 4 | 4 |
| `Scrap metal` | 1 | 1 |
| `Shoe` | 3 | 2 |
| `Single-use carrier bag` | 11 | 10 |
| `Six pack rings` | 1 | 1 |
| `Spread tub` | 1 | 1 |
| `Squeezable tube` | 2 | 2 |
| `Styrofoam piece` | 19 | 11 |
| `Tissues` | 4 | 4 |
| `Toilet tube` | 1 | 1 |
| `Tupperware` | 1 | 1 |
| `Unlabeled litter` | 103 | 59 |
| `Wrapping paper` | 5 | 4 |

## Split: `test`

| Metric | Value |
|---|---|
| Images | 150 |
| Annotations | 469 |
| Multi-object images | 94 |
| Background images (no annotation) | 0 |
| Corrupt images | 0 |
| Missing label files | 0 |
| Invalid annotation files | 0 |

### Class Instance Distribution

| Class | Instances | Images containing class |
|---|---|---|
| `Aerosol` | 0 | 0 |
| `Aluminium blister pack` | 0 | 0 |
| `Aluminium foil` | 8 | 3 |
| `Battery` | 0 | 0 |
| `Broken glass` | 0 | 0 |
| `Carded blister pack` | 0 | 0 |
| `Cigarette` | 80 | 34 |
| `Clear plastic bottle` | 29 | 24 |
| `Corrugated carton` | 5 | 4 |
| `Crisp packet` | 1 | 1 |
| `Disposable food container` | 2 | 2 |
| `Disposable plastic cup` | 8 | 8 |
| `Drink can` | 49 | 22 |
| `Drink carton` | 8 | 6 |
| `Egg carton` | 0 | 0 |
| `Foam cup` | 2 | 1 |
| `Foam food container` | 1 | 1 |
| `Food Can` | 3 | 2 |
| `Food waste` | 0 | 0 |
| `Garbage bag` | 5 | 2 |
| `Glass bottle` | 7 | 6 |
| `Glass cup` | 0 | 0 |
| `Glass jar` | 0 | 0 |
| `Magazine paper` | 1 | 1 |
| `Meal carton` | 1 | 1 |
| `Metal bottle cap` | 14 | 6 |
| `Metal lid` | 0 | 0 |
| `Normal paper` | 8 | 6 |
| `Other carton` | 13 | 9 |
| `Other plastic` | 29 | 18 |
| `Other plastic bottle` | 2 | 2 |
| `Other plastic container` | 0 | 0 |
| `Other plastic cup` | 0 | 0 |
| `Other plastic wrapper` | 21 | 14 |
| `Paper bag` | 2 | 2 |
| `Paper cup` | 6 | 5 |
| `Paper straw` | 0 | 0 |
| `Pizza box` | 0 | 0 |
| `Plastic bottle cap` | 22 | 22 |
| `Plastic film` | 32 | 24 |
| `Plastic glooves` | 0 | 0 |
| `Plastic lid` | 4 | 4 |
| `Plastic straw` | 8 | 7 |
| `Plastic utensils` | 3 | 3 |
| `Polypropylene bag` | 0 | 0 |
| `Pop tab` | 28 | 14 |
| `Rope - strings` | 3 | 3 |
| `Scrap metal` | 0 | 0 |
| `Shoe` | 0 | 0 |
| `Single-use carrier bag` | 7 | 6 |
| `Six pack rings` | 0 | 0 |
| `Spread tub` | 2 | 2 |
| `Squeezable tube` | 0 | 0 |
| `Styrofoam piece` | 12 | 8 |
| `Tissues` | 5 | 4 |
| `Toilet tube` | 0 | 0 |
| `Tupperware` | 0 | 0 |
| `Unlabeled litter` | 36 | 23 |
| `Wrapping paper` | 2 | 2 |

---

> **Note:** This report is generated automatically. Do not delete source images based on this report alone — review flagged files manually before taking action.