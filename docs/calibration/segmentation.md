# Segmentation-metric calibration report

Generated 2026-09-14 by `hatch run calibration:report` at framework commit `14ba43e`.
Framework stack: monai 1.6.0, pyiqa 0.1.15.post2, scipy 1.18.1, torch 2.14.0, numpy 2.5.3. Reference packages: medpy 0.5.2, surface-distance 0.1, panopticapi git 7bb4655548f9, boundary-iou-api vendored mask_to_boundary, git 37d25586a677.

Authority, in order: (1) the metric's definition applied by hand to a fixture small enough for the derivation to fit in a few lines — the derivation is what a reviewer checks; (2) the implementation the community cites, run on the same fixture with the same definitional variant; (3) MONAI's own test cases through this framework's loader and adapters. Δ = framework − hand value. Every framework value went through the public path (NIfTI → `load_pair` → `MetricRegistry` → evaluator → record).

## dice

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F1_dice | 0.833333 | Dice = 2*60/(72+72) = 120/144 (IoU would be 60/84 = 0.714). \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Dice 1945; Taha & Hanbury 2015, BMC Med Imaging 15:29 Eq. (6)* | 0.833333 — MedPy `dc` — voxel Dice | 0.833333 | -1.99e-08 | — |
| F1_aniso_dice | 0.833333 | Dice is spacing-invariant: 120/144. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Dice 1945; Taha & Hanbury 2015, BMC Med Imaging 15:29 Eq. (6)* | 0.833333 — MedPy `dc` — voxel Dice | 0.833333 | -1.99e-08 | — |
| F2_dice | 0.992248 | Dice = 2*64/(65+64) = 128/129. \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Dice 1945; Taha & Hanbury 2015, BMC Med Imaging 15:29 Eq. (6)* | 0.992248 — MedPy `dc` — voxel Dice | 0.992248 | -3.70e-09 | — |
| F6_dice_label2 | 0.750000 | Label 2: \|GT\| = \|Pred\| = 2*4*4 = 32, overlap W 3:6 -> 2*4*3 = 24 -> Dice 48/64. Slice counts: slices 1,2 each v_pred = v_gt = 16, tp = 12; slices 0,3 empty on both sides under label 2 -> is_empty, counts 0. Mask() (any non-zero): the identical label-1 block adds 4 to each volume and 4 to tp -> 56/72.<br>*Dice 1945; Taha & Hanbury 2015, BMC Med Imaging 15:29 Eq. (6)* | 0.750000 — MedPy `dc` — voxel Dice | 0.750000 | +0.00e+00 | — |
| F6_dice_any_label | 0.777778 | Label 2: \|GT\| = \|Pred\| = 2*4*4 = 32, overlap W 3:6 -> 2*4*3 = 24 -> Dice 48/64. Slice counts: slices 1,2 each v_pred = v_gt = 16, tp = 12; slices 0,3 empty on both sides under label 2 -> is_empty, counts 0. Mask() (any non-zero): the identical label-1 block adds 4 to each volume and 4 to tp -> 56/72.<br>*Dice 1945; Taha & Hanbury 2015, BMC Med Imaging 15:29 Eq. (6)* | 0.777778 — MedPy `dc` — voxel Dice | 0.777778 | +1.32e-08 | — |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_dice_gt_empty | 0.000000 | 0.000000 |
| F3_dice_pred_empty | 0.000000 | 0.000000 |
| F3_dice_both_empty | — | — |

Seeded random blobs — framework vs MedPy (same variant):

| case | spacing | framework | MedPy | Δ |
|---|---|---|---|---|
| F7_dice_1x1x1 | (1.0, 1.0, 1.0) | 0.534254 | 0.534254 | +2.91e-08 |
| F7_dice_1x1x3 | (1.0, 1.0, 3.0) | 0.408493 | 0.408493 | +1.04e-08 |
| F7_dice_2.5x0.8x0.8 | (2.5, 0.8, 0.8) | 0.505044 | 0.505044 | -2.14e-09 |

MONAI 1.6.0 test cases through `Mask()`/`Raw()` and the adapters:

| case | MONAI expected | framework | source |
|---|---|---|---|
| monai_dice_case_1 | 0.800000 | 0.800000 | MONAI 1.6.0 tests/metrics/test_compute_meandice.py TEST_CASE_1 |
| monai_dice_case_12_gt_empty | 0.000000 | 0.000000 | MONAI 1.6.0 tests/metrics/test_compute_meandice.py TEST_CASE_12 (ignore_empty=False) |
| monai_dice_case_11_both_empty | — | — | MONAI 1.6.0 tests/metrics/test_compute_meandice.py TEST_CASE_11 (ignore_empty=False) — MONAI expects 1.0; asserted as None per spec D3 |

## hausdorff95

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F1_hd95 | 1.000000 | P95 of 72 values with 12 ones (top 16.7 %) = 1 in both directions -> max 1.0. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)* | — (no same-variant implementation) | 1.000000 | +0.00e+00 | MedPy hd95 (P95 over concatenated distances): 1.000000<br>surface-distance robust_hausdorff(95) (area-weighted): 1.000000 |
| F1_hd | 1.000000 | Plain HD = max distance = 1.0. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)* | 1.000000 — MedPy `hd` — max over directed maxima (only at percentile=None) | 1.000000 | +0.00e+00 | — |
| F1_aniso_hd95 | 3.000000 | Every non-zero distance is one W step = 3 mm -> 3.0. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)* | — (no same-variant implementation) | 3.000000 | +0.00e+00 | MedPy hd95 (P95 over concatenated distances): 3.000000<br>surface-distance robust_hausdorff(95) (area-weighted): 3.000000 |
| F2_hd | 5.000000 | HD = max(0, 5) = 5.0. \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)* | 5.000000 — MedPy `hd` — max over directed maxima (only at percentile=None) | 5.000000 | +0.00e+00 | — |
| F2_hd95 | 0.000000 | P95 of [0]*56 + [5]: position 0.95*56 = 53.2 lies among the zeros -> 0; max(0, 0) = 0.0 (the percentile=95 mutation check: 5.0 vs 0.0). \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)* | — (no same-variant implementation) | 0.000000 | +0.00e+00 | MedPy hd95 (P95 over concatenated distances): 0.000000<br>surface-distance robust_hausdorff(95) (area-weighted): 0.000000 |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_hausdorff95_gt_empty | — | — |
| F3_hausdorff95_pred_empty | — | — |
| F3_hausdorff95_both_empty | — | — |

Seeded random blobs — framework vs MedPy (same variant):

| case | spacing | framework | MedPy | Δ |
|---|---|---|---|---|
| F7_hausdorff95_1x1x1 | (1.0, 1.0, 1.0) | 5.000000 | 5.000000 | +0.00e+00 |
| F7_hausdorff95_1x1x3 | (1.0, 1.0, 3.0) | 9.000000 | 9.000000 | +0.00e+00 |
| F7_hausdorff95_2.5x0.8x0.8 | (2.5, 0.8, 0.8) | 6.788225 | 6.788225 | +7.46e-08 |

MONAI 1.6.0 test cases through `Mask()`/`Raw()` and the adapters:

| case | MONAI expected | framework | source |
|---|---|---|---|
| monai_hd_identical | 0.000000 | 0.000000 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[0], expected index 1 (euclidean, directed=False) |
| monai_hd_shift_111 | 1.732051 | 1.732051 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[1], expected index 1 (euclidean, directed=False) |
| monai_hd_r33_shift_1 | 1.000000 | 1.000000 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[2], expected index 1 (euclidean, directed=False) |
| monai_hd_r20_vs_r40 | 20.223748 | 20.223749 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[3], expected index 1 (euclidean, directed=False) |
| monai_hd_pred_empty | — | — | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[4], expected index 1 (euclidean, directed=False) — MONAI expects inf; asserted as None per spec D3 |
| monai_hd_gt_empty | — | — | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[5], expected index 1 (euclidean, directed=False) — MONAI expects inf; asserted as None per spec D3 |
| monai_hd95_r20_vs_r40 | 20.099751 | 20.099751 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[6], expected index 1 (euclidean, directed=False) (percentile 95) |
| monai_hd_spacing_shift_111 | 2.267157 | 2.267157 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[7], expected index 1 (euclidean, directed=False) (spacing (0.85, 1.2, 0.9)) |
| monai_hd_spacing_r15_vs_r30 | 15.625940 | 15.625940 | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[8], expected index 1 (euclidean, directed=False) (spacing (0.85, 1.2, 0.9)) |
| monai_hd_both_empty | — | — | MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES_NANS[0] — MONAI expects nan; asserted as None per spec D3 |

## assd

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F1_assd | 0.166667 | ASSD = (12*1 + 12*1)/(72 + 72) = 24/144. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, ASSD (Eq. 25)* | 0.166667 — MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion | 0.166667 | +4.97e-09 | surface-distance ASD gt→pred (area-weighted): 0.188160<br>surface-distance ASD pred→gt (area-weighted): 0.188160 |
| F1_aniso_assd | 0.500000 | (12*3 + 12*3)/144 = 0.5. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, ASSD (Eq. 25)* | 0.500000 — MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion | 0.500000 | +0.00e+00 | surface-distance ASD gt→pred (area-weighted): 0.303784<br>surface-distance ASD pred→gt (area-weighted): 0.303784 |
| F2_assd | 0.044248 | Symmetric: (0*56 + 0*56 + 5)/(56 + 57) = 5/113. \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, ASSD (Eq. 25)* | 0.044248 — MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion | 0.044248 | -1.65e-10 | surface-distance ASD gt→pred (area-weighted): 0.000000<br>surface-distance ASD pred→gt (area-weighted): 0.093997 |
| F2_asd_directed | 0.087719 | Directed pred->gt: 5/57 (the symmetric=True mutation check). \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Taha & Hanbury 2015, BMC Med Imaging 15:29, ASSD (Eq. 25)* | 0.087719 — MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion | 0.087719 | +6.54e-10 | surface-distance ASD gt→pred (area-weighted): 0.000000<br>surface-distance ASD pred→gt (area-weighted): 0.093997 |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_assd_gt_empty | — | — |
| F3_assd_pred_empty | — | — |
| F3_assd_both_empty | — | — |

Seeded random blobs — framework vs MedPy (same variant):

| case | spacing | framework | MedPy | Δ |
|---|---|---|---|---|
| F7_assd_1x1x1 | (1.0, 1.0, 1.0) | 1.124637 | 1.124637 | -1.26e-07 |
| F7_assd_1x1x3 | (1.0, 1.0, 3.0) | 1.822560 | 1.822560 | +6.17e-09 |
| F7_assd_2.5x0.8x0.8 | (2.5, 0.8, 0.8) | 1.095254 | 1.095254 | +3.94e-08 |

MONAI 1.6.0 test cases through `Mask()`/`Raw()` and the adapters:

| case | MONAI expected | framework | source |
|---|---|---|---|
| monai_assd_identical | 0.000000 | 0.000000 | MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[0], expected index 0 (symmetric=True, euclidean) |
| monai_assd_r33_shift_1 | 0.350217 | 0.350217 | MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[2], expected index 0 (symmetric=True, euclidean) |
| monai_assd_r20_vs_r40 | 15.117741 | 15.117742 | MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[3], expected index 0 (symmetric=True, euclidean) |
| monai_assd_pred_empty | — | — | MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[6], expected index 0 (symmetric=True, euclidean) — MONAI expects inf; asserted as None per spec D3 |
| monai_assd_spacing | 0.495100 | 0.495100 | MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[8], expected index 0 (symmetric=True, euclidean) (spacing (0.85, 1.2, 0.9)) |

## nsd

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F1_nsd | 1.000000 | tau = 1 voxel: every surface voxel is within 1 -> 144/144. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Nikolov et al. 2018 (arXiv:1809.04430), voxel-surface variant as in MONAI compute_surface_dice* | — (no same-variant implementation) | 1.000000 | +0.00e+00 | surface-distance surface_dice@1 (area-weighted, author impl.): 1.000000 |
| F1_aniso_nsd | 0.833333 | tau = 1.0 mm: 12 + 12 surface voxels at 3 mm fail -> 120/144. \|GT\| = \|Pred\| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). Symmetric for pred->gt (W=9 -> W=8).<br>*Nikolov et al. 2018 (arXiv:1809.04430), voxel-surface variant as in MONAI compute_surface_dice* | — (no same-variant implementation) | 0.833333 | -1.99e-08 | surface-distance surface_dice@1 (area-weighted, author impl.): 0.904805 |
| F2_nsd | 0.991150 | tau = 1: (56 + 56)/(56 + 57) = 112/113. \|GT\| = 64, \|Pred\| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3)).<br>*Nikolov et al. 2018 (arXiv:1809.04430), voxel-surface variant as in MONAI compute_surface_dice* | — (no same-variant implementation) | 0.991150 | -3.69e-09 | surface-distance surface_dice@1 (area-weighted, author impl.): 0.989446 |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_nsd_gt_empty | 0.000000 | 0.000000 |
| F3_nsd_pred_empty | 0.000000 | 0.000000 |
| F3_nsd_both_empty | — | — |

MONAI 1.6.0 test cases through `Mask()`/`Raw()` and the adapters:

| case | MONAI expected | framework | source |
|---|---|---|---|
| monai_nsd_2d_shift_10px | 0.770461 | 0.770461 | MONAI 1.6.0 tests/metrics/test_surface_dice.py test_tolerance_euclidean_distance (no spacing), expected_res0[0, 1] (foreground class) |
| monai_nsd_3d_shift_10vox | 0.680827 | 0.680827 | MONAI 1.6.0 tests/metrics/test_surface_dice.py test_tolerance_euclidean_distance_3d, expected_res0[0, 1] (foreground class) |

## panoptic_quality

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F4_pq | 0.300000 | Kirillov 2019 Eq. 1: PQ = sum_TP IoU / (\|TP\| + 0.5\|FP\| + 0.5\|FN\|). GT1 vs Pred1: overlap 12, union 20, IoU 0.6 > 0.5 -> TP; GT2 unmatched -> FN; Pred3 unmatched -> FP. PQ = 0.6 / (1 + 0.5 + 0.5) = 0.3 (SQ 0.6, RQ 0.5). MONAI adds smooth_numerator=1e-6 to the denominator -> 0.6 / 2.000001 = 0.29999985 (0.29999986 in float32); tolerance 1e-6.<br>*Kirillov et al. 2019, CVPR, Eq. 1* | 0.300000 — panopticapi `pq_compute_single_core` — background as stuff, things-only PQ | 0.300000 | -1.37e-07 | — |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_panoptic_quality_gt_empty | 0.000000 | 0.000000 |
| F3_panoptic_quality_pred_empty | 0.000000 | 0.000000 |
| F3_panoptic_quality_both_empty | — | — |

MONAI 1.6.0 test cases through `Mask()`/`Raw()` and the adapters:

| case | MONAI expected | framework | source |
|---|---|---|---|
| monai_pq_func_case_2 | 0.250000 | 0.250000 | MONAI 1.6.0 tests/metrics/test_compute_panoptic_quality.py TEST_FUNC_CASE_2 |
| monai_pq_func_case_3_sq | 0.600000 | 0.600000 | MONAI 1.6.0 tests/metrics/test_compute_panoptic_quality.py TEST_FUNC_CASE_3 |
| monai_pq_func_case_4_rq_remap | 0.750000 | 0.750000 | MONAI 1.6.0 tests/metrics/test_compute_panoptic_quality.py TEST_FUNC_CASE_4 |

## boundary_iou

| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |
|---|---|---|---|---|---|---|
| F5_biou_D_shift | 0.295775 | Band width: 0.02*\|(24, 40, 40)\| = 1.229 mm < max(spacing) = 3 mm -> floor 3.0 mm. Inside an axis-aligned box the EDT to background is the perpendicular distance to the nearest face. D=2 and D=5 slabs lie 3 mm from background along D -> whole slab in the band (576 each); on D=3,4 (6 mm from background along D) only the in-plane ring of width 3, 24^2 - 18^2 = 252 each. \|band\| = 1152 + 504 = 1656 for GT and both predictions. D-shift: Pred band = slabs D=3,6 + rings D=4,5. Intersection D=3 ring&slab 252, D=4 ring&ring 252, D=5 slab&ring 252 = 756; union 2*1656 - 756 = 2556 -> 756/2556. W-shift: slabs D=2,5 intersect over W 9:32 -> 24*23 = 552 each; on D=3,4 within H 8:32 x W 9:32 the 6 ring rows give 6*23 = 138 and the other 18 rows share W in {9,10,30,31} -> 72; 210 each. Total 1104 + 420 = 1524; union 3312 - 1524 = 1788 -> 1524/1788. With the old 1.0 mm floor the band was the width-1 in-plane ring only (92 per slab, 368 total): D-shift 276/460 = 0.6, W-shift 184/552 = 0.333 -> the 3 mm error scored better (review 3.9).<br>*Cheng et al. 2021, CVPR, Sec. 3; 3D physical band is this framework's extension* | — (no same-variant implementation) | 0.295775 | +0.00e+00 | — |
| F5_biou_W_shift | 0.852349 | Band width: 0.02*\|(24, 40, 40)\| = 1.229 mm < max(spacing) = 3 mm -> floor 3.0 mm. Inside an axis-aligned box the EDT to background is the perpendicular distance to the nearest face. D=2 and D=5 slabs lie 3 mm from background along D -> whole slab in the band (576 each); on D=3,4 (6 mm from background along D) only the in-plane ring of width 3, 24^2 - 18^2 = 252 each. \|band\| = 1152 + 504 = 1656 for GT and both predictions. D-shift: Pred band = slabs D=3,6 + rings D=4,5. Intersection D=3 ring&slab 252, D=4 ring&ring 252, D=5 slab&ring 252 = 756; union 2*1656 - 756 = 2556 -> 756/2556. W-shift: slabs D=2,5 intersect over W 9:32 -> 24*23 = 552 each; on D=3,4 within H 8:32 x W 9:32 the 6 ring rows give 6*23 = 138 and the other 18 rows share W in {9,10,30,31} -> 72; 210 each. Total 1104 + 420 = 1524; union 3312 - 1524 = 1788 -> 1524/1788. With the old 1.0 mm floor the band was the width-1 in-plane ring only (92 per slab, 368 total): D-shift 276/460 = 0.6, W-shift 184/552 = 0.333 -> the 3 mm error scored better (review 3.9).<br>*Cheng et al. 2021, CVPR, Sec. 3; 3D physical band is this framework's extension* | — (no same-variant implementation) | 0.852349 | +0.00e+00 | — |
| F5_biou_2d | 0.333333 | Diagonal 56.57 px * 0.02 = 1.13 -> band 1 px. Ring of a 20x20 square: 2*20 + 2*18 = 76 px. Shifted ring shares rows 10 and 29 over cols 11..29 -> 2*19 = 38; the vertical sides do not coincide. Union 76 + 76 - 38 = 114 -> 38/114 = 1/3 (mask IoU would be 380/420 = 0.905).<br>*Cheng et al. 2021, CVPR, Sec. 3; 3D physical band is this framework's extension* | 0.333333 — boundary-iou-api `mask_to_boundary` (vendored) — 2D only | 0.333333 | +0.00e+00 | — |

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_boundary_iou_gt_empty | 0.000000 | 0.000000 |
| F3_boundary_iou_pred_empty | 0.000000 | 0.000000 |
| F3_boundary_iou_both_empty | — | — |

## vs

Empty-mask policy (spec D3):

| case | expected | framework |
|---|---|---|
| F3_vs_gt_empty | 0.000000 | 0.000000 |
| F3_vs_pred_empty | 0.000000 | 0.000000 |
| F3_vs_both_empty | — | — |

¹ The same metric under a different definitional variant (spec D10), reported for orientation and never used in an equality assertion: MedPy `hd95` takes the 95th percentile over the *concatenated* directed distances (MONAI/this framework: max of the two directed percentiles); DeepMind's `surface-distance` weights surface elements by area (MONAI/this framework: one surface voxel = one element). Boundary IoU's 3D physical band is this framework's extension of Cheng et al. and has a hand value only.
