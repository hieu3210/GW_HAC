# Reproduction package: Distortion-Bounded Linguistic Representation for Interpretable Multigranular Clustering

This repository contains the code, the data and the archived per-split results needed to reproduce every table,
figure and number of the paper

> *Distortion-Bounded Linguistic Representation for Interpretable Multigranular Clustering*, submitted to IEEE
> Transactions on Emerging Topics in Computational Intelligence (TETCI-2026-3160, 16 November 2026).
> Confirmatory protocol registered at OSF, doi:[10.17605/OSF.IO/KN54P](https://doi.org/10.17605/OSF.IO/KN54P).

The manuscript is under double-blind review: do not link this package from the review version. Section, table and
figure numbers below follow the TETCI manuscript (Roman table numbers). The study, the data and the archived results
are the same as for the earlier version of the paper; only the text changed, so the archive and its checksums are
unchanged.

## Quick start

```sh
sh reproduce.sh setup          # Python 3.14 virtual environment with the pinned packages + ExKMC (needs internet)
sh reproduce.sh                # verify, test, numerics, tables, check, tetci   (about 2 minutes)
```

The default run ends with two summaries:

- `66/66 checks passed` (`tools/check_outputs.py`): every table of Section V-C is rebuilt from the archived per-split
  results and compared byte for byte with the archive; the numerical verification of Section V-A reproduces Table I,
  Fig. 1 and its in-text numbers; the Iris example reproduces Table IV.
- `66/66 TETCI checks passed (6 with a printing erratum)` (`tools/check_tetci.py`): every other number printed in
  Sections V-A to V-D (the Gaussian decomposition example, Tables II, III and V, certificates, descriptions, noise,
  Spambase, the price on real features, Fig. 2, running times) is compared with the regenerated outputs. Six printed
  numbers are off by one unit in the last digit (see "Printing errata" below).

Two further stages each take about 30 seconds and recompute results from the data:

```sh
sh reproduce.sh spot           # rerun all development scripts on Iris and the confirmatory script on one dataset
sh reproduce.sh selection      # redo the confirmatory dataset selection offline from the archived OpenML metadata
```

The full recomputation (`rerun-dev`, `rerun-confirm`, `rerun-timing`) takes several hours with 8 processes. Its
output goes to `results/GW-HAC_v2/`, separate from the archive in `results/archive/`.

## The method in brief

A hedge algebra (HA) generates the linguistic values of a feature (*Very Little Large*, ...) from two generators
(*Small*, *Large*) and two hedges (*Very*, *Little*). Two parameters fix all of them:

- θ, the position of the neutral word *Medium*;
- α, the measure of the hedge *Little*.

The fuzziness intervals of the words of length ℓ partition [0, 1] into 2^ℓ nested cells. Mapping a value to the
semantically quantifying value (SQM) of its cell is the **HA quantizer**. The paper:

1. Characterizes the distortion of this quantizer by high-resolution theory.
   - The optimal θ solves a balance equation and lies between the median and ½.
   - Any α ≠ ½ has a slower exponential rate.
   - The vocabulary has a closed-form asymptotic price relative to optimal quantizers.
2. Applies the quantizer in **GW-HAC**. GW-HAC is a k-means-type clustering whose data and prototypes are vectors of
   words, one per feature. Its descriptions at all coarser levels are obtained by dropping hedges and come with
   per-sample certificates.

Per feature, (θ, α) is calibrated on the training data in one of two ways:

- **HA-grid** minimizes the training distortion over a 19×19 grid, at calibration level 4;
- **HA-θ** sets α = ½ and θ to the closed-form minimizer.

## Layout

| Path | Content |
|---|---|
| `reproduce.sh` | driver; `sh reproduce.sh <stage>` (stages listed at the top of the file) |
| `edge_hac/v2/` | experiment scripts of the paper (60–76), the shared library `v2lib.py`, the analysis protocol `protocol_v2.json`, unit tests |
| `edge_hac/v2/paper_tools/` | scripts that turn results into the paper's figures, Table 1, the Iris table, the timing numbers and the supplement |
| `edge_hac/v2/docs/` | `PREREG_E7_FILLED.md` (the document registered at OSF), its template, the hash manifest of the code frozen before the confirmatory run, and the frozen `run_v2.sh` |
| `edge_hac/edgehac/`, `haclustering/` | HA arithmetic (fuzziness measure, SQM, semantization), word-domain fitting, fuzzy-linguistic baselines |
| `edge_hac/experiments/` | three helper modules imported by `v2lib.py` (interval vocabularies, Beta and quantile families) |
| `data/`, `edge_hac/data/` | all datasets (see `DATASETS.md`) with `MANIFEST.json` files and `DATA_SHA256SUMS.txt` |
| `edge_hac/data/validation2/` | the 126 confirmatory datasets, the OpenML metadata snapshot of 6 November 2026, `SELECTION.json` and `SELECTION_LOCK.json` |
| `results/archive/` | the per-split results (`raw/`, 815 files), the tables and the logs of the runs reported in the paper, with `SHA256SUMS.txt` |
| `reference/figures/` | Figs. 1 and 2 as submitted |
| `tools/check_tetci.py` | comparison with the numbers printed in the TETCI manuscript (stage `tetci`) |
| `tools/` | ExKMC installer, output checks, raw-result comparison, dataset inventory |

## Where each result comes from

All commands are run by `reproduce.sh`. The paths in the last column are relative to `out/` after `tables` and
`numerics` have run, or to `results/archive/`.

| TETCI manuscript | Producing script(s) | Output |
|---|---|---|
| Sec. V-A, Table I, Fig. 1, in-text numbers | `v2/paper_tools/numerics_sec6.py` | `tables/tab1_medium.tex`, `figures/theory_verification.pdf`, `tables/numerics_sec6.json` |
| Sec. V-A, four Gaussian clusters (Proposition 6) | `numerics_sec6.py` (`decomposition_example`) | `numerics_sec6.json` (`decomposition`) |
| Table II (development and preliminary datasets) | `v2/60_dev_factorial.py` → `v2/69_dev_analyze.py` | `tables/v2_baselines_dev.tex`, `v2_baselines_val11.tex`, `v2_dev_summary_*.json` (`GWHAC3_vs`) |
| Factorial design (Sec. V-B, V-C "Clustering accuracy") | `60`, `69` | `tables/v2_factorial_dev.csv`, `v2_dev_summary_dev.json` (`factor_effects`) |
| Table III, "Medium and hedge asymmetry" | `v2/61_dev_theory.py`, `v2/61b_price_fix.py` → `69` | `tables/v2_theory_features_dev.tex`, `v2_decomposition_dev.tex`, `v2_certificates_dev.tex` |
| "Vocabulary price" (44–59% on near-continuous features) | `61b` | raw `v2price_*.csv`, computed in `tools/check_tetci.py` |
| "Objective decomposition" (slope −1.82, T₂ < 0 in 85%) | `61` | raw `v2dec_*.csv`, computed in `tools/check_tetci.py` |
| "Assignment certificates" (Theorem 4, Corollary 2, m*) | `61` → `69` | `tables/v2_certificates_dev.tex`, `v2_granularity_dev.tex` |
| Table IV (Iris) | `v2/paper_tools/worked_example.py iris` | `worked_example_iris.txt` |
| "Linguistic descriptions" (fidelity, tree depth) | `v2/62_dev_interp.py` → `69` | `tables/v2_interp_dev.tex` |
| "Noise and preprocessing" (Spambase, log/rank) | `v2/64_dev_noise.py`, `v2/63_dev_skew.py` → `69` | `tables/v2_noise_dev.tex`, `v2_skew_per_dataset_val11.csv` |
| "Runtime" (Sec. V-C, V-D) | `v2/75_timing.py` → `v2/paper_tools/timing_numbers.py` | `timing_numbers.txt`, `tables/timing_summary_by_dataset.csv` |
| Margin δ and sample size n (Sec. V-B) | `v2/70_power.py` | `tables/power.json` |
| Dataset selection (Sec. V-B) | `v2/71_select_validation.py --listing <snapshot>` | `edge_hac/data/validation2/SELECTION.json` |
| Table V, confirmatory text | `v2/73_confirm_run.py` → `v2/74_confirm_analyze.py` | `tables/confirm_table.tex`, `confirm.json`, `confirm_per_dataset.csv` |
| Fig. 2 | `v2/paper_tools/fig_confirm.py` | `figures/confirm_scatter.pdf` |
| (not part of the TETCI submission) supplementary tables of the earlier version | `v2/paper_tools/make_supp.py` | `supplement.tex` |

## The confirmatory study

The order of operations was enforced by the code:

1. **Selection.** `71` selected the datasets from OpenML **metadata only** and wrote `PREREG_E7_FILLED.md`. That
   document fixes the selection rules, δ, n, the hypotheses and the analysis.
2. **Registration.** The document was registered at OSF before any data file was downloaded.
3. **Download.** `72` refuses to download unless the OSF id is given and the document and the selection are unchanged
   (both are hashed in `SELECTION_LOCK.json`).
4. **Analysis.** `73` refuses to run unless the lock records the OSF id.

`sh reproduce.sh verify` checks these hashes. It also checks the code against the manifest frozen before the
confirmatory run, `edge_hac/v2/docs/LOCK_MANIFEST_v2.2.sha256`. Two differences are expected and are reported:

- `SELECTION_LOCK.json` gained the OSF id and the download times;
- `run_v2.sh` gained the `timing` stage after the confirmatory run. The frozen copy is `docs/run_v2.sh.v2.2_lock`.

`75_timing.py`, `76_timing_analyze.py` and `paper_tools/` were added after the confirmatory run. They do not change
any confirmatory result.

Of the 127 registered datasets, 126 were downloaded. OpenML id 1048 failed and, as registered, was not replaced.

**Do not run `run_v2.sh select` or `run_v2.sh fetch` in this folder.** They would overwrite the registered selection
files. Use `sh reproduce.sh selection` instead: it reruns the selection in a sandbox under `out/selection/`.

## Notes on exact reproduction

- **Determinism.** All splits, initializations and resamples are seeded (`protocol_v2.json`). On the reference machine
  (Apple M1 Pro, macOS 27, Python 3.14.7, NumPy 2.4.6, scikit-learn 1.9.0; `results/archive/environment.json`),
  recomputed per-split files are byte-identical to the archive. Other BLAS libraries or CPU architectures may change
  the last digits of k-means results and, rarely, a tie in an assignment.
- **`v2feat_*.csv`, column `mse_lloyd_tr`.**
  - The archived files come from the first development run, which used an iterative 1-D Lloyd quantizer as the
    reference for the price.
  - That reference was later replaced by an exact dynamic-programming quantizer, and `61b_price_fix.py` recomputed the
    price column used in Table III (`v2price_*.csv`).
  - Rerunning `61` with the current code therefore changes only this column. `tools/compare_raw.py` reports it as
    `same*`.
- **Timing.** Running times depend on the machine. The archived times were measured sequentially on one core of the
  reference machine.
  - `76_timing_analyze.py` groups by (dataset, N, F, K). Because F is counted after removing constant columns, it can
    vary between splits, so a few datasets are counted twice (n = 28 / 133 instead of 25 / 126).
  - The numbers in the paper are per dataset and come from `paper_tools/timing_numbers.py`.
- **Section V-A.** The script that produced the submitted Fig. 1 was not archived. `numerics_sec6.py` re-implements
  Section V-A from its specification:
  - midpoint quadrature with 4·10⁵ nodes (8·10⁵ for the α panel);
  - Lloyd's algorithm from the companding start;
  - for "HA, optimal", the best (θ, α) on a grid of step 0.005;
  - four Gaussian clusters: N = 3000, means uniform in [0.2, 0.8]⁴, SD 0.06, clipped, HA-θ, k-means partition
    (this configuration is now stated in the manuscript).

  It reproduces Table I exactly and every in-text number of Section V-A to the printed precision.
- **ExKMC.** ExKMC 0.0.3 is a source package whose Cython extension does not build on recent Python versions.
  `tools/install_exkmc.py` installs it with the pure-Python splitters shipped in the same package; this changes the
  speed, not the result. Without ExKMC, the IMM and ExKMC columns are skipped.

## Printing errata of the submitted manuscript

`tools/check_tetci.py` accepts a printed number that differs by one unit in its last digit and lists it. All six come
from rounding an already rounded value; none changes a conclusion. Correct them in the revision:

| Where | Printed | Correct (from the archive) |
|---|---|---|
| Table III, α̂ⱼ ∈ [0.45, 0.55], level 4 | 59% | 58% (0.5845) |
| Table III, sign agreement, level 2 | 72% | 71% (0.7146) |
| Table III, price of HA-grid calibrated at ℓ, level 4 | 1.87 | 1.86 (1.8647) |
| Sec. V-C, "At level 5, 75% were certified one level down" | 75% | 76% (0.7552) |
| Sec. V-C and V-D, median total runtime of HA-grid GW-HAC | 65 ms | 66 ms (65.5 ms) |
| Sec. V-C, informative subset, mean GW-HAC − FL-U8 | −0.019 | −0.018 (−0.01846) |

Two in-text numbers use a different family than Table III, which a reader may not expect: the mean log₂ slope −1.82 is
the absolute gap |J_ℓ − J_N| of HA-θ, and "T₂ negative in 85%" counts per-split HA-grid results.

## Datasets

`DATASETS.md` lists every dataset with N, F, K, source and SHA-256. It covers 25 development datasets (UCI Machine
Learning Repository and OpenML), 11 preliminary datasets and 126 confirmatory datasets (OpenML). All are public; the
copies here are provided for reproducibility under the terms of their sources.

## Licence and citation

Code: MIT licence (`LICENSE`). The datasets remain under the terms of their sources (`DATASETS.md`). If you use this package, please cite the paper above and the OSF registration.
