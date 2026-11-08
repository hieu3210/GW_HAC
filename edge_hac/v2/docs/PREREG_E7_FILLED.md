# Pre-registration — GW-HAC confirmatory validation (E7)

*Generated 2026-11-06T21:05:54 by `v2/71_select_validation.py` from this template.*

*Upload this file unchanged to OSF and obtain a time-stamp **before** running `v2/72_fetch_validation.py`. The fetch
script refuses to run if this file differs from SHA-256 `see SELECTION_LOCK.json`, which is recorded in
`data/validation2/SELECTION_LOCK.json`.*

## 1. Purpose
This study confirms, on OpenML datasets never used by the authors, the following claims of the revised GW-HAC paper:
- non-inferiority to the matched 8-label fuzzy-linguistic clustering (FL-U8);
- non-inferiority to k-means;
- equivalence of interval and nearest-value assignment;
- two predictions of the high-resolution theory.

There is no user study.

## 2. Methods (frozen; code commit 3dba7f5439c33c7c)

| Item | Setting |
|---|---|
| Proposed | GW-HAC at level 3, with interval assignment and SQM representatives. Each feature has its own (θ_j, α_j), taken from the 19×19 grid on [0.05, 0.95]² by training distortion at level 4. Feature weights are uniform. The model is the best of 5 initializations (1 deterministic + 4 k-means++), using Algorithm 1 of the paper. |
| Primary comparator | FL-U8: k-means (n_init = 10) on 8 uniform Ruspini triangular memberships per feature (8F dimensions). |
| Secondary comparators | k-means on the min–max features (n_init = 10). GW-nw: nearest SQM value among the 15 words with ≤ 2 hedges, θ = α = 1/2. |
| K | Number of classes of the default target. |
| Preprocessing | Non-numeric columns are dropped at download. On the training half: median imputation, then removal of constant columns, then min–max with the training range, clipped to [0, 1]. |
| Splits | 5 random half-splits per dataset (3 if N·F > 10⁶). Split r uses the permutation seed 1000 + r (`edgehac.datasets.split`). |
| Held-out assignment | Quantize with the training parameters, then assign to the nearest prototype. |
| Metric | Held-out ARI against the class labels, averaged over splits per dataset. |
| Secondary analysis (descriptive) | GW-HAC, FL-U8 and k-means repeated with log1p preprocessing: log1p(x − min) is applied, before min–max, to training features with min ≥ 0 and skewness > 2. It is reported as median differences with 95 % CI; no decision is taken. |

## 3. Dataset selection (OpenML metadata only)
A dataset is included if all of the following hold:
1. It is active. Only the latest active version of each name is used.
2. It does not match the exclusion patterns: generated or derived copies, image-pixel data, or the 22 name patterns in `protocol_v2.json`.
3. 2 ≤ K ≤ 20, and the majority class is ≤ 90 %.
4. 100 ≤ N ≤ 50 000 and 4 ≤ F_numeric ≤ 100, and at least half of the non-target columns are numeric (non-numeric columns are dropped at download; the development datasets all have F ≥ 4 and are all numeric).
5. Missing cells are ≤ 5 %.
6. It is none of the 43 datasets used in the earlier study, nor a version of any of them (exclusion list in `protocol_v2.json`).
7. It is not a copy of a used dataset: its (N, F, K) differs from those of every used dataset, and its name family
   (first token of the lower-cased name, split at `_ - . ( )` or space) is not the family of a used dataset.
8. Duplicates and families. Among the remaining datasets, a dataset with the same (N, K, majority-class size) as one
   with a lower OpenML id is removed, and the two ids listed in `protocol_v2.json → selection.exclude_ids_v2_1`
   (copies identified by inspection, with the reason recorded) are removed. Then **one dataset per name family** is kept, drawn with
   `numpy.random.default_rng(20261106)` over the families in alphabetical order (e.g. one of the FOREX series).
9. The metadata snapshot used for the selection is archived in `data/validation2/OPENML_LISTING.csv`, and its SHA-256 is
   recorded in `SELECTION_LOCK.json`.

**Sampling.**
- Qualifying datasets: 157.
- Ordering: ascending OpenML id.
- If more than n_max = 127 qualify, a simple random sample of n_max is drawn with `numpy.random.default_rng(20261106)`.
- Selected: 127 datasets. The list and its SHA-256 are in Appendix B.

A dataset that fails to download, has no numeric column, or has fewer than 2 classes after preprocessing is excluded
and reported. It is **not replaced**.

## 4. Margin and sample size (from the development runs, script 70)
- **δ = 0.020.** δ is the median, over the 25 development datasets, of the across-split SD of FL-U8 ARI (0.0182), rounded up to 0.005 with floor 0.02.
- **n = 110, n_max = ⌈1.15 n⌉ = 127.** n is the smallest value on the grid 40, 50, …, 200 that reaches simulated power ≥ 0.80 for H1, under these settings:
  - Each simulated study draws n values with replacement from the development differences d_i = mean_split(ARI_GW-HAC − ARI_FL-U8), centred to median 0.
  - It adds δ and applies the one-sided Wilcoxon signed-rank test at 0.025/3 = 0.0083, the smallest Holm threshold.
  - 4000 replicates, seed 20261106.
- **Formula not used.** The normal-approximation formula ⌈1.05·(z₀.₉₇₅ + z₀.₈)²·σ_d²/δ²⌉ with σ_d = 0.0953 would give 188. It is not used, because the development differences are heavy-tailed.
- **Analysis mode: test.**
  - "test" if at least n datasets are selected.
  - Otherwise "estimation": Family A is reported as median differences with 95 % CI only, without accept or reject decisions.

## 5. Hypotheses and decision rules
The per-dataset difference is d = ARI(GW-HAC) − ARI(comparator).

**Family A — confirmatory, Holm procedure, one-sided family-wise α = 0.025.**
- Order the three p-values ascending and compare them with 0.025/3, 0.025/2 and 0.025.
- Stop at the first non-rejection.
- The p-value of H3 is the larger of its two one-sided p-values.

| | Null hypothesis | Test | Claim if rejected |
|---|---|---|---|
| H1 | median(d_FL-U8) ≤ −δ | Wilcoxon signed-rank on d + δ, alternative "greater" | GW-HAC is non-inferior to FL-U8 |
| H2 | median(d_KM) ≤ −δ | as H1 | non-inferior to k-means |
| H3 | \|median(d_GW-nw)\| ≥ δ | TOST: two one-sided Wilcoxon tests, on d + δ ("greater") and d − δ ("less") | equivalent to GW-nw within ±δ |

**Sensitivity analyses of Family A (secondary).**
- They are reported with the same Holm procedure, and they do not change the Family A decisions above.
- **S1.** Excludes datasets on which every primary method (GW-HAC, FL-U8, k-means, GW-nw) has mean ARI < 0.02, i.e. labels carrying no cluster structure.
- **S2.** Excludes OpenML ids 700–999, the collection of regression problems binarized at the median, whose labels are artificial.
- The number of datasets removed is reported for each.

**Family B — theory predictions (Holm, α = 0.05).**

| | Prediction | Test |
|---|---|---|
| P1 | The objective gap decays faster than first order. The statistic is the per-dataset OLS slope b of log₂(\|J_ℓ − J_N\|/J_N + 10⁻⁶) on ℓ ∈ {2, 3, 4, 5}, computed on the k-means partition of the training half with the HA-grid vocabulary. The mean slope is < −1 (a first-order term would give −1 per level; a second-order term gives −2). | one-sided one-sample t-test of mean(b) < −1 |
| P2 | Hedge asymmetry vanishes with the level. For each dataset, compare the share of features whose α̂_j, grid-calibrated at level ℓ̄, lies in [0.45, 0.55], at ℓ̄ = 5 versus ℓ̄ = 2. The prediction is that the share is larger at ℓ̄ = 5. | one-sided sign test across datasets (ties dropped) |

**Reported for every hypothesis:**
- median paired difference;
- bootstrap 95 % CI (10 000 resamples of datasets, seed 20261106);
- p-value, n and the number of wins.

**Fixed descriptive outputs:**
- per-dataset ARI table;
- terms T1, T2, T3 of the decomposition;
- the α̂ shares by level.

## 6. Deviations
- Any deviation from this document is listed in the paper with its reason.
- No analysis is added to Family A or B after the time-stamp.

## Appendix A — excluded names and ids
See `protocol_v2.json` → `selection.exclude_names`, `selection.exclude_ids`. Candidates removed by these rules:
- 46208 Banknote (used name)
- 1462 banknote-authentication (used name)
- 43466 Bank-Note-Authentication-UCI (used name)
- 46913 blood-transfusion-service-center (used name)
- 44593 blood-transfusion-service-center_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44594 blood-transfusion-service-center_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44595 blood-transfusion-service-center_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44596 blood-transfusion-service-center_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44597 blood-transfusion-service-center_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 13 breast-cancer (used name)
- 42900 breast-cancer-coimbra (used name)
- 23499 breast-cancer-dropped-missing-attributes-values (used name)
- 43658 Breast-cancer-gene-expression---CuMiDa (used name)
- 43687 Breast-cancer-prediction (used name)
- 46591 Breast_Cancer_Wisconsin (used name)
- 43757 Breast-Cancer-Wisconsin-(Prognostic)-Data-Set (used name)
- 15 breast-w (used name)
- 40975 car (used name)
- 45746 cardiotocography (used name)
- 46921 diabetes (used name)
- 45219 diabetes_0 (used name)
- 45220 diabetes_1 (used name)
- 45229 diabetes_10 (used name)
- 43903 Diabetes-130-Hospitals_(Fairlearn) (used name)
- 46922 Diabetes130US (used name)
- 44578 Diabetes130US_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44579 Diabetes130US_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44580 Diabetes130US_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44581 Diabetes130US_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44582 Diabetes130US_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 45221 diabetes_2 (used name)
- 45222 diabetes_3 (used name)
- 45223 diabetes_4 (used name)
- 45224 diabetes_5 (used name)
- 45225 diabetes_6 (used name)
- 45226 diabetes_7 (used name)
- 45227 diabetes_8 (used name)
- 45228 diabetes_9 (used name)
- 46254 Diabetes_Dataset (used name)
- 43341 Diabetes-Dataset-2019 (used name)
- 791 diabetes_numeric (used name)
- 46869 diabetes_risk_prediction_dataset (used name)
- 44223 Diabetes(scikit-learn) (used name)
- 46614 Ecoli (used name)
- 46618 ECOLI70 (used name)
- 1471 eeg-eye-state (used id)
- 1030 ERA (used name)
- 1035 ESL (used name)
- 1476 gas-drift (used name)
- 1477 gas-drift-different-concentrations (used name)
- 42848 glass (used name)
- 43750 Glass-Classification (used name)
- 46334 glass_clean (used name)
- 1478 har (used id)
- 59 ionosphere (used name)
- 44344 Iris (used name)
- 1029 LEV (used name)
- 45736 libras (used name)
- 299 libras_move (used name)
- 44125 MagicTelescope (used name)
- 44397 MagicTelescope_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44398 MagicTelescope_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44399 MagicTelescope_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44400 MagicTelescope_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44401 MagicTelescope_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 1020 mfeat-karhunen (used name)
- 43938 nursery (used name)
- 980 optdigits (used name)
- 41197 ozone (used name)
- 301 ozone_level (used name)
- 1487 ozone-level-8hr (used id)
- 44548 ozone-level-8hr_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44549 ozone-level-8hr_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44550 ozone-level-8hr_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44551 ozone-level-8hr_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44552 ozone-level-8hr_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 1021 page-blocks (used name)
- 1488 parkinsons (used name)
- 4540 ParkinsonSpeechDatasetwithMultipleTypesofSoundRecordings (used name)
- 42176 parkinson-speech-uci (used name)
- 4531 parkinsons-telemonitoring (used name)
- 42857 pendigits (used name)
- 44127 phoneme (used name)
- 44407 phoneme_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44408 phoneme_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44409 phoneme_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44410 phoneme_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44411 phoneme_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 46952 qsar-biodeg (used name)
- 44598 qsar-biodeg_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44599 qsar-biodeg_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44600 qsar-biodeg_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44601 qsar-biodeg_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44602 qsar-biodeg_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 42858 satimage (used name)
- 1499 seeds (used name)
- 47040 seeds_dataset (used name)
- 42860 segment (used name)
- 44633 segment_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44634 segment_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44635 segment_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44636 segment_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44637 segment_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 46258 sonar (used name)
- 44 spambase (used id)
- 44095 spambase_reproduced (used name)
- 1028 SWD (used name)
- 40499 texture (used id)
- 40474 thyroid-allbp (used name)
- 40475 thyroid-allhyper (used name)
- 40476 thyroid-allhypo (used name)
- 40477 thyroid-allrep (used name)
- 40497 thyroid-ann (used name)
- 40478 thyroid-dis (used name)
- 47589 thyroid-disease (used name)
- 40682 thyroid-new (used name)
- 42864 vehicle (used name)
- 1242 vehicleNorm (used name)
- 44153 vehicle_reproduced (used name)
- 44553 vehicle_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44554 vehicle_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44555 vehicle_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44556 vehicle_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44557 vehicle_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 357 vehicle_sensIT (used name)
- 46262 Vehicle-Trips (used name)
- 47154 vowel (used name)
- 1526 wall-robot-navigation (used name)
- 47155 waveform (used name)
- 979 waveform-5000 (used name)
- 4551 WaveformDatabaseGenerator (used name)
- 1510 wdbc (used name)
- 40983 wilt (used id)
- 47041 Wine (used name)
- 45737 yeast (used name)
- 1103 yeast_gene (used name)
- 316 yeast_ml8 (used name)
- 44468 yeast_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44469 yeast_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44470 yeast_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44471 yeast_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)
- 44472 yeast_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True (used name)

## Appendix B — selected OpenML ids
- 446 prnn_crabs v1 (N=200, F_num=6, K=2)
- 481 biomed v1 (N=209, F_num=7, K=2)
- 717 rmftsa_ladata v2 (N=508, F_num=10, K=2)
- 721 pwLinear v2 (N=200, F_num=10, K=2)
- 725 bank8FM v2 (N=8192, F_num=8, K=2)
- 733 machine_cpu v2 (N=209, F_num=6, K=2)
- 735 cpu_small v3 (N=8192, F_num=12, K=2)
- 750 pm10 v2 (N=500, F_num=7, K=2)
- 752 puma32H v2 (N=8192, F_num=32, K=2)
- 756 autoPrice v2 (N=159, F_num=15, K=2)
- 757 meta v2 (N=528, F_num=19, K=2)
- 770 strikes v2 (N=625, F_num=6, K=2)
- 788 triazines v2 (N=186, F_num=60, K=2)
- 803 delta_ailerons v1 (N=7129, F_num=5, K=2)
- 818 diggle_table_a2 v2 (N=310, F_num=7, K=2)
- 820 chatfield_4 v2 (N=235, F_num=12, K=2)
- 825 boston_corrected v2 (N=506, F_num=17, K=2)
- 833 bank32nh v2 (N=8192, F_num=32, K=2)
- 840 autoHorse v2 (N=205, F_num=17, K=2)
- 843 house_8L v2 (N=22784, F_num=8, K=2)
- 847 wind v2 (N=6574, F_num=14, K=2)
- 871 pollen v2 (N=3848, F_num=5, K=2)
- 880 mu284 v2 (N=284, F_num=10, K=2)
- 886 no2 v2 (N=500, F_num=7, K=2)
- 890 cloud v2 (N=108, F_num=6, K=2)
- 897 colleges_aaup v2 (N=1161, F_num=13, K=2)
- 907 chscase_census4 v2 (N=400, F_num=7, K=2)
- 915 plasma_retinol v2 (N=315, F_num=10, K=2)
- 925 visualizing_galaxy v2 (N=323, F_num=4, K=2)
- 954 spectrometer v2 (N=531, F_num=100, K=2)
- 957 braziltourism v2 (N=412, F_num=4, K=2)
- 976 JapaneseVowels v2 (N=9961, F_num=14, K=2)
- 1004 synthetic_control v2 (N=600, F_num=60, K=2)
- 1046 mozilla4 v1 (N=15545, F_num=5, K=2)
- 1048 jEdit_4.2_4.3 v1 (N=369, F_num=8, K=2)
- 1049 pc4 v1 (N=1458, F_num=37, K=2)
- 1050 pc3 v1 (N=1563, F_num=37, K=2)
- 1054 mc2 v1 (N=161, F_num=39, K=2)
- 1061 ar4 v1 (N=107, F_num=29, K=2)
- 1063 kc2 v1 (N=522, F_num=21, K=2)
- 1067 kc1 v1 (N=2109, F_num=21, K=2)
- 1100 PopularKids v1 (N=478, F_num=6, K=3)
- 1121 badges2 v1 (N=294, F_num=7, K=2)
- 1167 pc1_req v1 (N=320, F_num=7, K=2)
- 1441 KungChi3 v1 (N=123, F_num=39, K=2)
- 1447 CastMetal1 v1 (N=327, F_num=37, K=2)
- 1448 KnuggetChase3 v1 (N=194, F_num=39, K=2)
- 1450 MindCave2 v1 (N=125, F_num=39, K=2)
- 1453 PieChart3 v1 (N=1077, F_num=37, K=2)
- 1459 artificial-characters v1 (N=10218, F_num=7, K=10)
- 1473 fertility v1 (N=100, F_num=9, K=2)
- 1475 first-order-theorem-proving v1 (N=6118, F_num=51, K=6)
- 1490 planning-relax v1 (N=182, F_num=12, K=2)
- 1496 ringnorm v1 (N=7400, F_num=20, K=2)
- 1498 sa-heart v1 (N=462, F_num=8, K=2)
- 1507 twonorm v1 (N=7400, F_num=20, K=2)
- 1508 user-knowledge v1 (N=403, F_num=5, K=5)
- 1511 wholesale-customers v1 (N=440, F_num=7, K=2)
- 1565 heart-h v3 (N=294, F_num=13, K=5)
- 1566 hill-valley v2 (N=1212, F_num=100, K=2)
- 1600 SPECTF v2 (N=267, F_num=44, K=2)
- 4538 GesturePhaseSegmentationProcessed v1 (N=9873, F_num=32, K=5)
- 40700 cars1 v1 (N=392, F_num=6, K=3)
- 40705 tokyo1 v1 (N=959, F_num=42, K=2)
- 40966 MiceProtein v4 (N=1080, F_num=77, K=8)
- 41027 jungle_chess_2pcs_raw_endgame_complete v1 (N=44819, F_num=6, K=3)
- 41146 sylvine v1 (N=5124, F_num=20, K=2)
- 41156 ada v1 (N=4147, F_num=48, K=2)
- 41430 DiabeticMellitus v1 (N=281, F_num=97, K=2)
- 41671 microaggregation2 v1 (N=20000, F_num=20, K=5)
- 41701 CPMP-2015-classification v1 (N=527, F_num=24, K=4)
- 41879 FOREX_gbpusd-hour-High v1 (N=43825, F_num=11, K=2)
- 41978 TuningSVMs v3 (N=156, F_num=80, K=2)
- 42585 penguins v1 (N=344, F_num=4, K=3)
- 43892 national-longitudinal-survey-binary v1 (N=4908, F_num=9, K=2)
- 43901 click_prediction_small v11 (N=39926, F_num=4, K=2)
- 43904 law-school-admission-bianry v3 (N=20800, F_num=5, K=2)
- 43905 ibm-employee-performance v3 (N=1470, F_num=24, K=2)
- 43925 eucalyptus v4 (N=4331, F_num=5, K=4)
- 44124 kdd_ipums_la_97-small v6 (N=5188, F_num=20, K=2)
- 44186 KDDCup09_upselling v6 (N=5128, F_num=34, K=2)
- 44227 South_Asian_Churn_dataset v1 (N=2000, F_num=9, K=2)
- 45028 california v7 (N=20634, F_num=8, K=2)
- 45604 dummy v1 (N=1000, F_num=6, K=2)
- 45615 appendicitis_test_edsa v3 (N=106, F_num=7, K=2)
- 45711 doa_bwin v3 (N=530, F_num=11, K=2)
- 46264 mabbob_ela_as_2d_classify v1 (N=1120, F_num=45, K=5)
- 46279 algerian_forest_fires v1 (N=243, F_num=13, K=2)
- 46298 electrical_grid_stability_simulated_classification v1 (N=10000, F_num=12, K=2)
- 46331 autos_clean v1 (N=205, F_num=15, K=6)
- 46371 Customer_Churn_Classification_Nominal v1 (N=35000, F_num=24, K=2)
- 46372 Multiclass_Classification_for_Corporate_Credit_Ratings v1 (N=5000, F_num=7, K=10)
- 46375 default-of-credit-card-clients v7 (N=30000, F_num=13, K=2)
- 46382 company_quality_and_valuation_finance v1 (N=719, F_num=9, K=6)
- 46416 Creditability-German-Credit-Data v1 (N=1000, F_num=20, K=2)
- 46420 Apple_Stock_Price_Trends_(2014-2023) v1 (N=2516, F_num=18, K=3)
- 46431 HMEQ_Data v3 (N=5960, F_num=10, K=2)
- 46467 IBRD_Loans_Classification v4 (N=9217, F_num=5, K=11)
- 46481 Financial-Risk-Assessment v1 (N=15000, F_num=10, K=3)
- 46505 total_score v1 (N=719, F_num=10, K=4)
- 46506 Credit_Score v1 (N=50000, F_num=21, K=3)
- 46551 Corporate_Credit v3 (N=2029, F_num=25, K=10)
- 46563 Loan_Approval_Status_Classification v2 (N=45000, F_num=13, K=2)
- 46667 imdb_genre_prediction v1 (N=800, F_num=7, K=2)
- 46808 news_channel v2 (N=20284, F_num=16, K=6)
- 46840 Estimation_of_Obesity_Levels v2 (N=2111, F_num=8, K=7)
- 46846 glioma_grading_clinical_and_mutation_features v2 (N=839, F_num=22, K=2)
- 46851 hepatitis_c_virus_hcv_for_egyptian_patients v3 (N=1385, F_num=28, K=4)
- 46863 Fetal_cardiotocography_dataset v3 (N=2126, F_num=36, K=10)
- 46868 Phishing_Legitimate_full v2 (N=10000, F_num=48, K=2)
- 46870 agriculture_dataset_karnataka v2 (N=3158, F_num=7, K=13)
- 46871 Advanced_IoT_Dataset v2 (N=30000, F_num=12, K=6)
- 46875 divorce_prediction v2 (N=170, F_num=54, K=2)
- 46876 depression_2020 v2 (N=1429, F_num=20, K=2)
- 46879 mental_health_detection v2 (N=540, F_num=14, K=4)
- 46880 air-quality-and-pollution-assessment v2 (N=5000, F_num=9, K=4)
- 46924 E-CommereShippingData v1 (N=10999, F_num=6, K=2)
- 46930 hazelnut-spread-contaminant-detection v1 (N=2400, F_num=30, K=2)
- 46932 heloc v3 (N=10459, F_num=23, K=2)
- 46941 maternal_health_risk v3 (N=1014, F_num=6, K=3)
- 46944 Mobile_Price v1 (N=2000, F_num=14, K=4)
- 46951 Pumpkin_Seeds v1 (N=2500, F_num=12, K=2)
- 46959 steel-plates-fault v4 (N=1941, F_num=25, K=7)
- 46960 students_dropout_and_academic_success v1 (N=4424, F_num=19, K=3)
- 46979 jm1 v3 (N=10885, F_num=21, K=2)
- 47150 CanadaFiresD v1 (N=1491, F_num=46, K=2)
- 48184 NovAsia_Property_Listing_Dedup_Synthetic v1 (N=4800, F_num=9, K=2)

SHA-256 of the id list: `ac010549dd209c78053a90c923811ec0f0a0e20429cc98af36fe2aa53f81b5ba`
