"""Exploratory GW-I (ADDENDUM_U4): interval semantization over the level-k Little words, k in {2, 3, 4}.

Arms: GW-I(1/2) (= dyadic equal-width grid), GW-I(cal) (per-feature (theta, alpha) from the H5 grid by training
interval-MSE at k = 4, shared across k), QW-I (nested training quantiles). Protocol of 22_gw_numeric.py
(EWA-HAC weights, best of 5, held-out ARI). Output: <out>/raw/gwi_<ds>.csv, <out>/raw/gwi_h4_<ds>.csv.

    python experiments/27_gw_interval.py --out ../results/GW-HAC_2026.10.27 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI

from edgehac import datasets as D
from edgehac import sqm
from edgehac.fit import fit_ewahac, semantize
from edgehac.whac import CustomVocab, best_of, predict_words, semantize_words

PROTO = json.loads((HERE / "configs" / "protocol_unified.json").read_text())
N_INIT = PROTO["GW"]["n_init"]
GRID = np.linspace(*PROTO["H5"]["grid_range"], PROTO["H5"]["grid"])
LEVELS = (2, 3, 4)
K_CAL = 4


def n_splits_for(N: int, F: int) -> int:
    P = PROTO["protocol_numeric"]
    return P["splits_if_NF_gt_1e6"] if N * F > 1_000_000 else P["splits"]


def level_cells(theta: float, alpha: float, k: int):
    """Level-k Little words in ascending order: SQM values, inner interval boundaries, codes (T2.1 partition)."""
    iv = sorted((sqm.interval(w, theta, alpha, "little"), sqm.nu(w, theta, alpha, "little"), w)
                for w in sqm.level_words(k))
    return np.array([c[1] for c in iv]), np.array([c[0][1] for c in iv[:-1]]), [c[2] for c in iv]


def interval_vocab(th: np.ndarray, al: np.ndarray, k: int) -> CustomVocab:
    cells = [level_cells(float(a), float(b), k) for a, b in zip(th, al)]
    return CustomVocab(np.stack([c[0] for c in cells]), np.stack([c[1] for c in cells]), [c[2] for c in cells])


def nested_quantile_vocab(S_tr: np.ndarray, k: int) -> CustomVocab:
    T = 2 ** k
    thr = np.stack([np.quantile(S_tr[:, j], np.arange(1, T) / T) for j in range(S_tr.shape[1])])
    return CustomVocab(np.tile((np.arange(T) + .5) / T, (S_tr.shape[1], 1)), thr)


def interval_mse(S: np.ndarray, voc: CustomVocab) -> np.ndarray:
    q = semantize_words(S, voc)
    return ((voc.V[np.arange(S.shape[1])[None, :], q] - S) ** 2).mean(0)


def calibrate(S_tr: np.ndarray):
    F = S_tr.shape[1]
    best, arg = np.full(F, np.inf), np.zeros((F, 2))
    for th in GRID:
        for al in GRID:
            v, b, _ = level_cells(th, al, K_CAL)
            m = ((v[np.searchsorted(b, S_tr, side="right")] - S_tr) ** 2).mean(0)
            near_half = (th - .5) ** 2 + (al - .5) ** 2
            better = (m < best - 1e-15) | ((np.abs(m - best) <= 1e-15) & (near_half < ((arg - .5) ** 2).sum(1)))
            best[better], arg[better] = m[better], (th, al)
    return arg[:, 0], arg[:, 1]


def consistency(vc, fc, qc, vf, ff, qf, w, parent_map):
    """H4 definitions: sample-level share pi(fine word) = coarse word; prototype-level after Hungarian matching."""
    F = qc.shape[1]
    fidx = np.arange(F)[None, :]
    samp = float(np.mean(parent_map[fidx, qf] == qc))
    nu_c = vc.V[fidx, fc.P]
    proj = parent_map[fidx, ff.P]
    nu_fp = vc.V[fidx, proj]
    C = np.sqrt(np.einsum("abf,f->ab", (nu_c[:, None] - nu_fp[None]) ** 2, w))
    r, c = linear_sum_assignment(C)
    return samp, float(np.mean(fc.P[r] == proj[c]))


def run_dataset(name: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / f"gwi_h4_{name}.csv").exists():
        return f"[skip] {name}"
    t0 = time.time()
    data = D.load(name)
    X, y = data["X"], data["y"]
    K = int(len(np.unique(y)))
    N, F = X.shape
    rows, h4 = [], []
    for sp in range(n_splits_for(N, F)):
        tr, te = D.split(N, sp)
        fit = fit_ewahac(X[tr], K, seed=sp)
        S_tr, S_te = semantize(X[tr], fit.lo, fit.hi), semantize(X[te], fit.lo, fit.hi)
        w = fit.weights / fit.weights.sum()
        th_c, al_c = calibrate(S_tr)
        arms = {"GW-I(1/2)": lambda k: interval_vocab(np.full(F, .5), np.full(F, .5), k),
                "GW-I(cal)": lambda k: interval_vocab(th_c, al_c, k),
                "QW-I": lambda k: nested_quantile_vocab(S_tr, k)}
        for arm, mk in arms.items():
            fits = {}
            for k in LEVELS:
                voc = mk(k)
                q_tr, q_te = semantize_words(S_tr, voc), semantize_words(S_te, voc)
                f = best_of(q_tr, voc, K, n_init=N_INIT, seed=sp, weights=w)
                fits[k] = (voc, f, q_te)
                rows.append(dict(dataset=name, split=sp, method=arm, k=k, T=2 ** k,
                                 ari=float(ARI(y[te], predict_words(q_te, voc, f.P, weights=w))),
                                 mse_te=float(interval_mse(S_te, voc).mean()),
                                 theta_mean=float(th_c.mean()) if arm == "GW-I(cal)" else np.nan,
                                 alpha_mean=float(al_c.mean()) if arm == "GW-I(cal)" else np.nan))
            for k in LEVELS[:-1]:
                vc, fc, qc = fits[k]
                vf, ff, qf = fits[k + 1]
                if arm == "QW-I":                     # cell t at level k+1 lies in cell t // 2 at level k
                    pm = np.tile(np.arange(2 ** (k + 1)) // 2, (F, 1))
                else:                                 # hedge projection pi: drop the outermost hedge
                    pm = np.stack([[vc.names[j].index(wd[1:]) for wd in vf.names[j]] for j in range(F)])
                samp, proto = consistency(vc, fc, qc, vf, ff, qf, w, pm)
                h4.append(dict(dataset=name, split=sp, method=arm, k=k, sample=samp, proto=proto))
    pd.DataFrame(rows).to_csv(raw / f"gwi_{name}.csv", index=False)
    pd.DataFrame(h4).to_csv(raw / f"gwi_h4_{name}.csv", index=False)
    return f"[done] {name}: N={N} F={F} K={K} in {time.time() - t0:.0f}s"


def _worker(a):
    try:
        return run_dataset(*a)
    except Exception:
        return f"[FAIL] {a[0]}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "GW-HAC_2026.10.27"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    names = a.only or list(D.available())
    order = sorted(names, key=lambda n: -D.load(n)["X"].size)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, (n, Path(a.out))) for n in order]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
