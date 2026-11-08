# Pre-registration — GW-HAC confirmatory validation (E7)

*Generated {{GENERATED}} by `v2/71_select_validation.py` from this template.*

*Upload this file unchanged to OSF and obtain a time-stamp **before** running `v2/72_fetch_validation.py`. The fetch
script refuses to run if this file differs from SHA-256 `{{SELF_SHA_NOTE}}`, which is recorded in
`data/validation2/SELECTION_LOCK.json`.*

## 1. Purpose
This study confirms, on OpenML datasets never used by the authors, the following claims of the revised GW-HAC paper:
- non-inferiority to the matched 8-label fuzzy-linguistic clustering (FL-U8);
- non-inferiority to k-means;
- equivalence of interval and nearest-value assignment;
- two predictions of the high-resolution theory.

There is no user study.

## 2. Methods (frozen; code commit {{CODE_HASH}})

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
2. It does not match the exclusion patterns: generated or derived copies, image-pixel data, or the {{N_PATTERNS}} name patterns in `protocol_v2.json`.
3. 2 ≤ K ≤ 20, and the majority class is ≤ 90 %.
4. 100 ≤ N ≤ 50 000 and 4 ≤ F_numeric ≤ 100, and at least half of the non-target columns are numeric (non-numeric columns are dropped at download; the development datasets all have F ≥ 4 and are all numeric).
5. Missing cells are ≤ 5 %.
6. It is none of the 43 datasets used in the earlier study, nor a version of any of them (exclusion list in `protocol_v2.json`).
7. It is not a copy of a used dataset: its (N, F, K) differs from those of every used dataset, and its name family
   (first token of the lower-cased name, split at `_ - . ( )` or space) is not the family of a used dataset.
8. Duplicates and families. Among the remaining datasets, a dataset with the same (N, K, majority-class size) as one
   with a lower OpenML id is removed, and the two ids listed in `protocol_v2.json → selection.exclude_ids_v2_1`
   (copies identified by inspection, with the reason recorded) are removed. Then **one dataset per name family** is kept, drawn with
   `numpy.random.default_rng({{SEL_SEED}})` over the families in alphabetical order (e.g. one of the FOREX series).
9. The metadata snapshot used for the selection is archived in `data/validation2/OPENML_LISTING.csv`, and its SHA-256 is
   recorded in `SELECTION_LOCK.json`.

**Sampling.**
- Qualifying datasets: {{N_QUALIFY}}.
- Ordering: ascending OpenML id.
- If more than n_max = {{N_MAX}} qualify, a simple random sample of n_max is drawn with `numpy.random.default_rng({{SEL_SEED}})`.
- Selected: {{N_SELECTED}} datasets. The list and its SHA-256 are in Appendix B.

A dataset that fails to download, has no numeric column, or has fewer than 2 classes after preprocessing is excluded
and reported. It is **not replaced**.

## 4. Margin and sample size (from the development runs, script 70)
- **δ = {{DELTA}}.** δ is the median, over the {{N_DEV}} development datasets, of the across-split SD of FL-U8 ARI ({{DELTA_RAW}}), rounded up to 0.005 with floor 0.02.
- **n = {{N_REQ}}, n_max = ⌈1.15 n⌉ = {{N_MAX}}.** n is the smallest value on the grid 40, 50, …, 200 that reaches simulated power ≥ 0.80 for H1, under these settings:
  - Each simulated study draws n values with replacement from the development differences d_i = mean_split(ARI_GW-HAC − ARI_FL-U8), centred to median 0.
  - It adds δ and applies the one-sided Wilcoxon signed-rank test at 0.025/3 = 0.0083, the smallest Holm threshold.
  - 4000 replicates, seed {{SIM_SEED}}.
- **Formula not used.** The normal-approximation formula ⌈1.05·(z₀.₉₇₅ + z₀.₈)²·σ_d²/δ²⌉ with σ_d = {{SIGMA_D}} would give {{N_FORMULA}}. It is not used, because the development differences are heavy-tailed.
- **Analysis mode: {{MODE}}.**
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
- bootstrap 95 % CI (10 000 resamples of datasets, seed {{BOOT_SEED}});
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
{{EXCLUDED_MATCHES}}

## Appendix B — selected OpenML ids
{{SELECTED_LIST}}

SHA-256 of the id list: `{{SEL_SHA}}`
