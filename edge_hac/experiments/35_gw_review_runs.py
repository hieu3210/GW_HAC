"""New runs requested in the internal review round (ADDENDUM_U7, B1-B7).

Output per dataset: raw/rev_<ds>.csv (held-out ARI/NMI), raw/revstab_<ds>.csv (description stability).

    python experiments/35_gw_review_runs.py --out ../results/GW-HAC_2026.10.27 --workers 8
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
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
from scipy.stats import beta as Beta
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

from edgehac import datasets as D
from edgehac.baselines_gw import memberships, ruspini_centres
from edgehac.fit import fit_ewahac, semantize
from edgehac.whac import CustomVocab, best_of, fit_words, predict_words, semantize_words


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GWI = _load("27_gw_interval")
NUM = _load("22_gw_numeric")
PROTO = json.loads((HERE / "configs" / "protocol_unified.json").read_text())
N_INIT, LEVEL, N_LAB = PROTO["GW"]["n_init"], 3, 100
BGRID = np.geomspace(0.25, 4.0, 19)
FL_GRID = [(T, pl) for T in (3, 5, 7, 9, 15) for pl in ("uniform", "quantile")]


# ---------------- B1: Beta-warped dyadic cells ----------------
def beta_cells(a: float, b: float, level: int = LEVEL):
    T = 2 ** level
    return Beta.ppf((np.arange(T) + .5) / T, a, b), Beta.ppf(np.arange(1, T) / T, a, b)


def beta_vocab(A_, B_, level: int = LEVEL) -> CustomVocab:
    cells = [beta_cells(a, b, level) for a, b in zip(A_, B_)]
    return CustomVocab(np.stack([c[0] for c in cells]), np.stack([c[1] for c in cells]))


def beta_calibrate(S_tr: np.ndarray, level: int = 4):
    F = S_tr.shape[1]
    best, arg = np.full(F, np.inf), np.ones((F, 2))
    for a in BGRID:
        for b in BGRID:
            v, thr = beta_cells(a, b, level)
            m = ((v[np.searchsorted(thr, S_tr, side="right")] - S_tr) ** 2).mean(0)
            near = (np.log(a)) ** 2 + (np.log(b)) ** 2
            better = (m < best - 1e-15) | ((np.abs(m - best) <= 1e-15) & (near < (np.log(arg) ** 2).sum(1)))
            best[better], arg[better] = m[better], (a, b)
    return arg[:, 0], arg[:, 1]


# ---------------- B2: nested quantile cells with in-cell means ----------------
def quantile_mean_vocab(S_tr: np.ndarray, level: int = LEVEL) -> CustomVocab:
    T, F = 2 ** level, S_tr.shape[1]
    thr = np.stack([np.quantile(S_tr[:, j], np.arange(1, T) / T) for j in range(F)])
    V = np.empty((F, T))
    for j in range(F):
        idx = np.searchsorted(thr[j], S_tr[:, j], side="right")
        edges = np.concatenate([[0.0], thr[j], [1.0]])
        for t in range(T):
            m = idx == t
            V[j, t] = S_tr[m, j].mean() if m.any() else 0.5 * (edges[t] + edges[t + 1])
        V[j] = np.maximum.accumulate(V[j] + np.arange(T) * 1e-12)       # keep representatives non-decreasing
    return CustomVocab(V, thr)


def fl_fit(S_tr, K, T, placement, seed):
    C = ruspini_centres(T) if placement == "uniform" else ruspini_centres(T, S_tr)
    Z = memberships(S_tr, C).reshape(len(S_tr), -1)
    km = KMeans(K, n_init=PROTO["FL"]["kmeans_n_init"], random_state=seed).fit(Z)
    return km, C


def fl_predict(km, C, S):
    return km.predict(memberships(S, C).reshape(len(S), -1))


def word_fit(S_tr, S_te, voc, K, seed, w):
    f = best_of(semantize_words(S_tr, voc), voc, K, n_init=N_INIT, seed=seed, weights=w)
    return f, predict_words(semantize_words(S_te, voc), voc, f.P, weights=w)


def stability(protos: list) -> float:
    """Mean share of identical (feature, word) entries between Hungarian-matched prototypes of two splits.
    ``protos`` holds (codes (K, F), values (K, F)) per split; matching uses the representative values."""
    out = []
    for (c1, v1), (c2, v2) in itertools.combinations(protos, 2):
        cost = ((v1[:, None, :] - v2[None]) ** 2).sum(2)
        r, c = linear_sum_assignment(cost)
        out.append(float(np.mean(c1[r] == c2[c])))
    return float(np.mean(out)) if out else np.nan


def run_dataset(name: str, out: Path) -> str:
    raw = out / "raw"
    if (raw / f"revstab_{name}.csv").exists():
        return f"[skip] {name}"
    t0 = time.time()
    data = D.load(name)
    X, y = data["X"], data["y"]
    K, (N, F) = int(len(np.unique(y))), X.shape
    rows, stab = [], {k: [] for k in ("GW-I", "GW-I level 1", "GW-I(1/2)", "BW-I(cal)", "FL-U T=7")}
    fidx = np.arange(F)[None, :]
    for sp in range(NUM.n_splits_for(N, F)):
        tr, te = D.split(N, sp)
        y_tr, y_te = y[tr], y[te]

        def add(method, lab, **extra):
            rows.append(dict(dataset=name, split=sp, method=method, ari=float(ARI(y_te, lab)), nmi=float(NMI(y_te, lab)),
                             **extra))

        fit = fit_ewahac(X[tr], K, seed=sp)
        S_tr, S_te = semantize(X[tr], fit.lo, fit.hi), semantize(X[te], fit.lo, fit.hi)
        w = fit.weights / fit.weights.sum()
        idx = NUM.labelled_subset(y_tr, N_LAB, sp)
        # ---- GW-I primary (reference for stability), dyadic grid ----
        th, al = GWI.calibrate(S_tr)
        voc = GWI.interval_vocab(th, al, LEVEL)
        f, _ = word_fit(S_tr, S_te, voc, K, sp, w)
        codes = np.array([[voc.names[j][f.P[k, j]] for j in range(F)] for k in range(K)])
        vals = voc.V[fidx, f.P]
        stab["GW-I"].append((codes, vals))
        stab["GW-I level 1"].append((np.vectorize(lambda s: s[-1])(codes), vals))
        half = GWI.interval_vocab(np.full(F, .5), np.full(F, .5), LEVEL)
        fh, _ = word_fit(S_tr, S_te, half, K, sp, w)
        stab["GW-I(1/2)"].append((fh.P.copy(), half.V[fidx, fh.P]))
        # ---- B1: Beta-warped dyadic cells ----
        a_, b_ = beta_calibrate(S_tr)
        bv = beta_vocab(a_, b_)
        fb, lab = word_fit(S_tr, S_te, bv, K, sp, w)
        add("BW-I(cal)", lab)
        stab["BW-I(cal)"].append((fb.P.copy(), bv.V[fidx, fb.P]))
        best, arg = -np.inf, (1.0, 1.0)
        for a in BGRID:
            for b in BGRID:
                vv = beta_vocab(np.full(F, a), np.full(F, b))
                sc = ARI(y_tr[idx], fit_words(semantize_words(S_tr, vv), vv, K, weights=w).labels[idx])
                if sc > best + 1e-12 or (abs(sc - best) <= 1e-12 and np.log(a) ** 2 + np.log(b) ** 2 < np.log(arg[0]) ** 2 + np.log(arg[1]) ** 2):
                    best, arg = sc, (a, b)
        _, lab = word_fit(S_tr, S_te, beta_vocab(np.full(F, arg[0]), np.full(F, arg[1])), K, sp, w)
        add("BW-I(lab100)", lab, a=float(arg[0]), b=float(arg[1]))
        # ---- B2: nested quantiles with in-cell means ----
        _, lab = word_fit(S_tr, S_te, quantile_mean_vocab(S_tr), K, sp, w)
        add("QW-I(mean)", lab)
        # ---- B3: fuzzy-linguistic with the same label budget; FL-U T=7 for stability ----
        best, cfg, fits = -np.inf, None, {}
        for T, pl in FL_GRID:
            km, C = fl_fit(S_tr, K, T, pl, sp)
            fits[(T, pl)] = (km, C)
            sc = ARI(y_tr[idx], km.labels_[idx])
            if sc > best + 1e-12:
                best, cfg = sc, (T, pl)
        km, C = fits[cfg]
        add("FL(lab100)", fl_predict(km, C, S_te), T=cfg[0], placement=cfg[1])
        km7, _ = fits[(7, "uniform")]
        cent = km7.cluster_centers_.reshape(K, F, 7)
        stab["FL-U T=7"].append((cent.argmax(2), (cent * np.linspace(0, 1, 7)).sum(2) / np.maximum(cent.sum(2), 1e-12)))
        # ---- B4: robust range ----
        lo, hi = np.percentile(X[tr], 1, axis=0), np.percentile(X[tr], 99, axis=0)
        hi = np.where(hi > lo, hi, lo + 1e-12)
        R_tr, R_te = semantize(X[tr], lo, hi), semantize(X[te], lo, hi)
        thr_, alr = GWI.calibrate(R_tr)
        _, lab = word_fit(R_tr, R_te, GWI.interval_vocab(thr_, alr, LEVEL), K, sp, w)
        add("GW-I(robust)", lab)
        km15, C15 = fl_fit(R_tr, K, 15, "uniform", sp)
        add("FL-U(robust)", fl_predict(km15, C15, R_te))
        add("KM(robust)", KMeans(K, n_init=10, random_state=sp).fit(R_tr).predict(R_te))
        # ---- B5: K - 1 and K + 1 ----
        for dK in (-1, 1):
            K2 = K + dK
            if K2 < 2:
                continue
            _, lab = word_fit(S_tr, S_te, voc, K2, sp, w)
            add(f"GW-I(K{dK:+d})", lab)
            kmK, CK = fl_fit(S_tr, K2, 15, "uniform", sp)
            add(f"FL-U(K{dK:+d})", fl_predict(kmK, CK, S_te))
            add(f"KM(K{dK:+d})", KMeans(K2, n_init=10, random_state=sp).fit(S_tr).predict(S_te))
        # ---- B7: FCM on the normalized features ----
        import skfuzzy
        cntr, *_ = skfuzzy.cluster.cmeans(S_tr.T, K, 2.0, error=1e-5, maxiter=300, seed=sp)
        u, *_ = skfuzzy.cluster.cmeans_predict(S_te.T, cntr, 2.0, error=1e-5, maxiter=300, seed=sp)
        add("FCM-S", u.argmax(0))
    pd.DataFrame(rows).to_csv(raw / f"rev_{name}.csv", index=False)
    pd.DataFrame([dict(dataset=name, method=m, stability=stability(v)) for m, v in stab.items()]).to_csv(
        raw / f"revstab_{name}.csv", index=False)
    return f"[done] {name}: N={N} F={F} in {time.time() - t0:.0f}s"


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
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    names = a.only or list(D.available())
    order = sorted(names, key=lambda n: -D.load(n)["X"].size)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, (n, out)) for n in order]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
