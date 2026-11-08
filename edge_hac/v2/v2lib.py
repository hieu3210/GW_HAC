"""Shared helpers of the GW-HAC v2 experiments (manuscript "High-Resolution Theory of Hedge-Algebra Quantization").

Everything is built on the original package: interval vocabularies (27_gw_interval.py), the word-domain Lloyd
iteration (edgehac.whac), Beta-warp / nested-quantile families (35_gw_review_runs.py) and the Ruspini fuzzy-linguistic
baseline (edgehac.baselines_gw).  Feature weights are uniform (1/F) throughout, as decided for v2.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # edge_hac/v2
EDGE = HERE.parent                                # edge_hac
REPO = EDGE.parent                                # GW_HAC_Reproduction
sys.path[:0] = [str(EDGE), str(REPO)]
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from edgehac import datasets as D
from edgehac.baselines_gw import memberships, ruspini_centres
from edgehac.whac import (CustomVocab, embed, fit_words, kmeanspp_prototypes, make_vocab, predict_words,
                          semantize_words, _nearest_word)

CFG = json.loads((HERE / "protocol_v2.json").read_text())
LEVELS = tuple(CFG["levels"])
MAIN_LEVEL = CFG["main_level"]
CAL_LEVEL = CFG["cal_level"]
GRID = np.linspace(*CFG["grid_range"], CFG["grid"])
N_INIT, MAX_ITER = CFG["n_init"], CFG["max_iter"]
RESULTS = REPO / "results" / "GW-HAC_v2"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EDGE / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GWI = _load("27_gw_interval")       # level_cells, interval_vocab, calibrate (cal level 4), consistency
REV = _load("35_gw_review_runs")    # beta_vocab, beta_calibrate, quantile_mean_vocab


# ---------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------
VAL11_DIR = EDGE / "data" / "validation"
VAL2_DIR = Path(os.environ.get("GWHAC_VAL2_DIR", str(EDGE / "data" / "validation2")))   # override for tests only


def dev_names():
    return list(D.available())                    # 25 development datasets


def val11_names():
    man = json.loads((VAL11_DIR / "MANIFEST.json").read_text())
    return [n for n, m in man.items() if "dropped" not in m and (VAL11_DIR / f"{n}.npz").exists()]


def load_any(name: str, group: str):
    if group == "dev":
        d = D.load(name)
        return d["X"].astype(float), d["y"]
    folder = VAL11_DIR if group == "val11" else VAL2_DIR
    z = np.load(folder / f"{name}.npz", allow_pickle=False)
    return z["X"].astype(float), z["y"].astype(np.int64)


def n_splits_for(N: int, F: int) -> int:
    return CFG["splits_if_NF_gt_1e6"] if N * F > 1_000_000 else CFG["splits"]


def prepare_split(X: np.ndarray, tr: np.ndarray, te: np.ndarray, mode: str = "minmax"):
    """Training-half preprocessing: median imputation, constant-column removal, then min-max (default),
    log1p for skewed non-negative features ("log1p"), or the training ECDF ("rank").  Returns S_tr, S_te, kept columns."""
    Xtr, Xte = X[tr].copy(), X[te].copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for A in (Xtr, Xte):
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = np.take(med, np.nonzero(bad)[1])
    keep = Xtr.max(0) > Xtr.min(0)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if mode == "log1p":
        from scipy.stats import skew
        sk = skew(Xtr, axis=0)
        mn = Xtr.min(0)
        sel = (mn >= 0) & (sk > CFG["skew_log_threshold"])
        Xtr[:, sel], Xte[:, sel] = np.log1p(Xtr[:, sel] - mn[sel]), np.log1p(np.maximum(Xte[:, sel] - mn[sel], 0))
    if mode == "rank":
        S_tr, S_te = np.empty_like(Xtr), np.empty_like(Xte)
        n = len(Xtr)
        for j in range(Xtr.shape[1]):
            srt = np.sort(Xtr[:, j])
            for A, B in ((Xtr, S_tr), (Xte, S_te)):
                lo_ = np.searchsorted(srt, A[:, j], side="left")
                hi_ = np.searchsorted(srt, A[:, j], side="right")
                B[:, j] = np.clip((lo_ + hi_) / (2.0 * n), 0, 1)
        return S_tr, S_te, keep
    lo, hi = Xtr.min(0), Xtr.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((Xtr - lo) / span, 0, 1), np.clip((Xte - lo) / span, 0, 1), keep


def uniform_w(F: int) -> np.ndarray:
    return np.full(F, 1.0 / F)


# ---------------------------------------------------------------------------------------------------------
# HA calibration: closed-form Medium (Theorem 2) and grid calibration at any level
# ---------------------------------------------------------------------------------------------------------
def theta_hat(S: np.ndarray, step: float = CFG["theta_hat_grid_step"]) -> np.ndarray:
    """Per-feature minimiser of the empirical Phi(theta) = theta^2 F(theta) + (1-theta)^2 (1-F(theta)) on
    [0.05, 0.95] (ties -> nearest 1/2)."""
    th = np.arange(CFG["grid_range"][0], CFG["grid_range"][1] + 1e-12, step)
    out = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        Fh = np.searchsorted(np.sort(S[:, j]), th, side="right") / len(S)
        phi = th ** 2 * Fh + (1 - th) ** 2 * (1 - Fh)
        best = np.flatnonzero(phi <= phi.min() + 1e-15)
        out[j] = th[best[np.argmin(np.abs(th[best] - .5))]]
    return out


def calibrate_grid(S_tr: np.ndarray, level: int = CAL_LEVEL, alpha_fixed=None, theta_fixed=None):
    """Grid calibration by training distortion (as GWI.calibrate, but at any level, optionally with alpha or
    theta fixed per feature).  Returns (theta_j, alpha_j)."""
    F = S_tr.shape[1]
    best, arg = np.full(F, np.inf), np.full((F, 2), .5)
    ths = [None] if theta_fixed is not None else GRID
    als = [None] if alpha_fixed is not None else GRID
    for th in ths:
        for al in als:
            if th is None or al is None:                         # per-feature fixed component
                for j in range(F):
                    t = theta_fixed[j] if th is None else th
                    a = alpha_fixed[j] if al is None else al
                    v, b, _ = GWI.level_cells(float(t), float(a), level)
                    m = ((v[np.searchsorted(b, S_tr[:, j], side="right")] - S_tr[:, j]) ** 2).mean()
                    nh = (t - .5) ** 2 + (a - .5) ** 2
                    if m < best[j] - 1e-15 or (abs(m - best[j]) <= 1e-15 and nh < ((arg[j] - .5) ** 2).sum()):
                        best[j], arg[j] = m, (t, a)
                continue
            v, b, _ = GWI.level_cells(th, al, level)
            m = ((v[np.searchsorted(b, S_tr, side="right")] - S_tr) ** 2).mean(0)
            nh = (th - .5) ** 2 + (al - .5) ** 2
            better = (m < best - 1e-15) | ((np.abs(m - best) <= 1e-15) & (nh < ((arg - .5) ** 2).sum(1)))
            best[better], arg[better] = m[better], (th, al)
    return arg[:, 0], arg[:, 1]


def hedge_count(names_j, level: int) -> np.ndarray:
    """H(x) = n_L(x) - n_V(x) for the ascending words of one feature."""
    return np.array([w[:-1].count("L") - w[:-1].count("V") for w in names_j], float)


def Q_hat(S: np.ndarray, theta: np.ndarray, level: int) -> np.ndarray:
    """Empirical Q_l(theta) of Proposition 5 at alpha = 1/2, per feature."""
    voc = GWI.interval_vocab(theta, np.full(S.shape[1], .5), level)
    q = semantize_words(S, voc)
    out = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        H = hedge_count(voc.names[j], level)[q[:, j]]
        left = S[:, j] < theta[j]
        out[j] = theta[j] ** 2 * np.mean(H * left) + (1 - theta[j]) ** 2 * np.mean(H * ~left)
    return out


# ---------------------------------------------------------------------------------------------------------
# Vocabulary families x representatives x assignment rules (factorial design E1)
# ---------------------------------------------------------------------------------------------------------
def _edges(thr_j):
    return np.concatenate([[0.0], thr_j, [1.0]])


def with_rep(voc, rep: str, S_tr: np.ndarray):
    """Replace the representatives of a vocabulary: "native" (unchanged), "mid" (cell midpoints) or "mean"
    (training cell means; empty cells -> midpoint; kept non-decreasing).  Cells (voc.thr) are unchanged."""
    if rep == "native":
        return voc
    F, T = voc.V.shape
    V = np.empty((F, T))
    for j in range(F):
        e = _edges(voc.thr[j])
        mid = 0.5 * (e[:-1] + e[1:])
        if rep == "mid":
            V[j] = mid
        else:
            idx = np.searchsorted(voc.thr[j], S_tr[:, j], side="right")
            s = np.bincount(idx, S_tr[:, j], T)
            c = np.bincount(idx, None, T)
            V[j] = np.where(c > 0, s / np.maximum(c, 1), mid)
            V[j] = np.maximum.accumulate(V[j] + np.arange(T) * 1e-12)
    return CustomVocab(V, voc.thr.copy(), voc.names)


def with_assign(voc, rule: str):
    """"interval": cells as given; "nearest": a value takes the word with the nearest representative."""
    if rule == "interval":
        return voc
    return CustomVocab(voc.V.copy(), 0.5 * (voc.V[:, 1:] + voc.V[:, :-1]), voc.names)


def family_vocab(family: str, level: int, S_tr: np.ndarray, cache: dict):
    """Native vocabulary of a family at a level; calibration is computed once per split and cached."""
    F = S_tr.shape[1]
    if family == "dyadic":
        return GWI.interval_vocab(np.full(F, .5), np.full(F, .5), level)
    if family == "HA-theta":
        if "theta_hat" not in cache:
            cache["theta_hat"] = theta_hat(S_tr)
        return GWI.interval_vocab(cache["theta_hat"], np.full(F, .5), level)
    if family == "HA-grid":
        if "grid" not in cache:
            cache["grid"] = GWI.calibrate(S_tr)               # 19 x 19, training distortion at level 4
        return GWI.interval_vocab(*cache["grid"], level)
    if family == "Beta":
        if "beta" not in cache:
            cache["beta"] = REV.beta_calibrate(S_tr)
        return REV.beta_vocab(*cache["beta"], level)
    if family == "Quantile":
        T = 2 ** level
        thr = np.stack([np.quantile(S_tr[:, j], np.arange(1, T) / T) for j in range(F)])
        thr = np.maximum.accumulate(thr + np.arange(T - 1)[None] * 1e-12, axis=1)
        e = np.concatenate([np.zeros((F, 1)), thr, np.ones((F, 1))], 1)
        return CustomVocab(0.5 * (e[:, :-1] + e[:, 1:]), thr)
    raise KeyError(family)


FACTORIAL = ([(fam, rep, rule) for fam in ("dyadic", "HA-theta", "HA-grid") for rep in ("native", "mid", "mean")
              for rule in ("interval", "nearest")]
             + [("Beta", rep, "interval") for rep in ("native", "mid", "mean")]
             + [("Quantile", rep, "interval") for rep in ("native", "mean")])
PRIMARY = ("HA-grid", "native", "interval")            # GW-HAC as published (SQM representatives, interval cells)


def tag(cfg) -> str:
    return "|".join(cfg)


# ---------------------------------------------------------------------------------------------------------
# Word-domain fitting with bookkeeping of iteration-cap hits (same result as edgehac.whac.best_of)
# ---------------------------------------------------------------------------------------------------------
def best_of_v2(q, voc, K, seed, w, n_init=N_INIT, max_iter=MAX_ITER):
    best = fit_words(q, voc, K, max_iter=max_iter, weights=w)
    hits = int(best.n_iter >= max_iter)
    nuX = embed(q, voc)
    rng = np.random.default_rng(seed)
    for _ in range(n_init - 1):
        f = fit_words(q, voc, K, init=kmeanspp_prototypes(nuX, voc, K, rng), max_iter=max_iter, weights=w)
        hits += int(f.n_iter >= max_iter)
        if f.J < best.J:
            best = f
    return best, hits


def word_model(S_tr, S_te, voc, K, seed, w):
    q_tr, q_te = semantize_words(S_tr, voc), semantize_words(S_te, voc)
    f, hits = best_of_v2(q_tr, voc, K, seed, w)
    return f, hits, q_tr, q_te, predict_words(q_te, voc, f.P, weights=w)


# ---------------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------------
def fl_uniform(S_tr, S_te, K, T, seed):
    """FL-U with T uniform Ruspini labels: k-means on the T*F membership vector (n_init from the protocol)."""
    C = ruspini_centres(T)
    Z_tr = memberships(S_tr, C).reshape(len(S_tr), -1)
    km = KMeans(K, n_init=CFG["kmeans_n_init"], random_state=seed).fit(Z_tr)
    return km, C, km.predict(memberships(S_te, C).reshape(len(S_te), -1))


def fl_win_values(S, T):
    """Centre of the winning (maximal-membership) uniform label: F-dimensional, same geometry as GW-HAC."""
    c = np.linspace(0, 1, T)
    return c[np.clip(np.rint(S * (T - 1)).astype(int), 0, T - 1)]


def fl_win(S_tr, S_te, K, T, seed):
    km = KMeans(K, n_init=CFG["kmeans_n_init"], random_state=seed).fit(fl_win_values(S_tr, T))
    return km.predict(fl_win_values(S_te, T))


def kmeans(S_tr, S_te, K, seed):
    km = KMeans(K, n_init=CFG["kmeans_n_init"], random_state=seed).fit(S_tr)
    return km, km.predict(S_te)


def gw_nw(S_tr, S_te, K, seed, w):
    """GW-nw: nearest SQM value among the 15 words with <= 2 hedges, theta = alpha = 1/2 (study arm "GW", s = 2)."""
    voc = make_vocab(.5, .5, S_tr.shape[1], 2, variant="little")
    return word_model(S_tr, S_te, voc, K, seed, w)[-1]


def threshold_trees(S_tr, S_te, K, seed):
    """IMM / ExKMC if the ExKMC library is installed (tools/install_exkmc.py); otherwise {}."""
    try:
        from ExKMC.Tree import Tree
    except Exception:
        return {}
    km = KMeans(K, n_init=10, random_state=seed).fit(S_tr)
    out = {}
    for kind, leaves in (("IMM", K), ("ExKMC", 2 * K)):
        t = Tree(k=K, max_leaves=leaves, base_tree="IMM", random_state=seed)
        t.fit(S_tr, km)
        out[kind] = (t.predict(S_te), t)
    return out


def tree_depth(tree, X):
    d = np.zeros(len(X))
    for i, x in enumerate(X):
        node, k = tree.tree, 0
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.value else node.right
            k += 1
        d[i] = k
    return float(d.mean())


# ---------------------------------------------------------------------------------------------------------
# Theory quantities
# ---------------------------------------------------------------------------------------------------------
def wdist(A, C, w, chunk=2048):
    out = np.empty((len(A), len(C)))
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = np.sqrt(np.einsum("nkf,f->nk", (A[i:i + chunk, None, :] - C[None]) ** 2, w))
    return out


def decomposition(S, labels, voc, w):
    """Proposition 7 for a fixed partition: returns JN, Jl, T1, T2, T3, N*D_hat (all with weights w)."""
    K = int(labels.max()) + 1
    F = S.shape[1]
    q = semantize_words(S, voc)
    v = embed(q, voc)
    e = v - S
    z = np.stack([S[labels == k].mean(0) if np.any(labels == k) else np.zeros(F) for k in range(K)])
    eb = np.stack([e[labels == k].mean(0) if np.any(labels == k) else np.zeros(F) for k in range(K)])
    vb = z + eb
    P = _nearest_word(vb, voc)
    nuP = voc.V[np.arange(F)[None, :], P]
    nk = np.bincount(labels, minlength=K)
    JN = float((w * (S - z[labels]) ** 2).sum())
    Jl = float((w * (v - nuP[labels]) ** 2).sum())
    T1 = float((w * (e - eb[labels]) ** 2).sum())
    T2 = float(2 * (w * (S - z[labels]) * e).sum())
    T3 = float((nk[:, None] * w * (vb - nuP) ** 2).sum())
    ND = float((w * e ** 2).sum())
    return dict(JN=JN, Jl=Jl, T1=T1, T2=T2, T3=T3, ND=ND, identity_err=abs(Jl - JN - T1 - T2 - T3))


def max_cell_error(voc):
    """r_{l,j}: largest |representative - value| over the cells of each feature."""
    F = voc.V.shape[0]
    lo = np.concatenate([np.zeros((F, 1)), voc.thr], 1)
    hi = np.concatenate([voc.thr, np.ones((F, 1))], 1)
    return np.max(np.maximum(voc.V - lo, hi - voc.V), 1)


def translation_certificate(S, Z, w, voc):
    """Theorem 4 for a numeric model (prototypes Z): data-dependent and static (rho > 4 rbar) conditions."""
    N, K = len(S), len(Z)
    dN = wdist(S, Z, w)
    g = dN.argmin(1)
    srt = np.sort(dN, 1)
    rho = srt[:, 1] - srt[:, 0] if K > 1 else np.full(N, np.inf)
    F = S.shape[1]
    fidx = np.arange(F)[None, :]
    Xw = voc.V[fidx, semantize_words(S, voc)]
    Pw = voc.V[fidx, semantize_words(Z, voc)]
    dW = wdist(Xw, Pw, w)
    E, Dl = Xw - S, Pw - Z
    dev = np.sqrt(np.einsum("nkf,f->nk", (E[:, None, :] - Dl[None]) ** 2, w))
    other = np.max(np.where(np.arange(K)[None] == g[:, None], -np.inf, dev), 1)
    safe = rho > dev[np.arange(N), g] + other
    rbar = float(np.sqrt((w * max_cell_error(voc) ** 2).sum()))
    safe_s = rho > 4 * rbar
    gt = dW.argmin(1)
    chg = gt != g
    return dict(g=g, g_trans=gt, cert=safe, cert_static=safe_s, changed=chg, rbar=rbar)


def match_labels(a, b, K):
    """Relabel ``a`` to best agree with ``b`` (Hungarian on the confusion matrix)."""
    C = np.zeros((K, K))
    np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(-C)
    m = np.arange(K)
    m[r] = c
    return m[a]


def certified_granularity(q_te, voc_fine, P, w, level, voc_coarse_fn):
    """Corollary 2 / Definition 2 for a fitted level-``level`` word model: per-sample m*(i) and, per m, the
    certified share, the observed keep rate and violations.  ``voc_coarse_fn(l)`` builds the level-l vocabulary."""
    F = q_te.shape[1]
    fidx = np.arange(F)[None, :]
    Xv, Pv = voc_fine.V[fidx, q_te], voc_fine.V[fidx, P]
    dW = wdist(Xv, Pv, w)
    g = dW.argmin(1)
    K = len(P)
    srt = np.sort(dW, 1)
    rho = srt[:, 1] - dW[np.arange(len(g)), g] if K > 1 else np.full(len(g), np.inf)
    mstar = np.zeros(len(g), int)
    alive = np.ones(len(g), bool)
    per_m = []
    for m in range(1, level):
        vc = voc_coarse_fn(level - m)
        qc, Pc = q_te >> m, P >> m
        gc = predict_words(qc, vc, Pc, weights=w)
        E, Dl = vc.V[fidx, qc] - Xv, vc.V[fidx, Pc] - Pv
        dev = np.sqrt(np.einsum("nkf,f->nk", (E[:, None, :] - Dl[None]) ** 2, w))
        other = np.max(np.where(np.arange(K)[None] == g[:, None], -np.inf, dev), 1)
        safe = rho > dev[np.arange(len(g)), g] + other
        alive &= safe
        mstar[alive] = m
        per_m.append(dict(m=m, cert=float(safe.mean()), keep=float(np.mean(gc == g)),
                          viol=int((safe & (gc != g)).sum())))
    return mstar, per_m


def lloyd_1d(x: np.ndarray, T: int) -> float:
    """EXACT optimal T-point scalar quantizer of the empirical distribution of x (MSE).  1-D k-means dynamic programme
    over the sorted unique values with divide and conquer on the monotone split points.  (Replaces the Lloyd iteration
    of the first v2 release, which lost levels on tied data; see 61b_price_fix.py.)"""
    u, cnt = np.unique(x, return_counts=True)
    U = len(u)
    if U <= T:
        return 0.0
    cs = np.concatenate([[0], np.cumsum(cnt)]).astype(float)
    c1 = np.concatenate([[0], np.cumsum(cnt * u)])
    c2 = np.concatenate([[0], np.cumsum(cnt * u * u)])

    def cost(j, i):
        n = cs[i] - cs[j]
        s_ = c1[i] - c1[j]
        return np.maximum(c2[i] - c2[j] - s_ * s_ / n, 0.0)

    prev = np.full(U + 1, np.inf)
    i = np.arange(1, U + 1)
    prev[1:] = np.maximum(c2[i] - c1[i] ** 2 / cs[i], 0.0)
    for k in range(2, T + 1):
        cur = np.full(U + 1, np.inf)
        stack = [(k, U, k - 1, U - 1)]
        while stack:
            lo, hi, olo, ohi = stack.pop()
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            js = np.arange(max(olo, k - 1), min(mid - 1, ohi) + 1)
            vals = prev[js] + cost(js, mid)
            b = int(np.argmin(vals))
            cur[mid] = vals[b]
            stack.append((lo, mid - 1, olo, int(js[b])))
            stack.append((mid + 1, hi, int(js[b]), ohi))
        prev = cur
    return float(prev[U] / cs[-1])


def vocab_mse(S, voc):
    q = semantize_words(S, voc)
    return ((voc.V[np.arange(S.shape[1])[None, :], q] - S) ** 2).mean(0)


# ---------------------------------------------------------------------------------------------------------
# Running over datasets
# ---------------------------------------------------------------------------------------------------------
def run_pool(fn, jobs, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import traceback

    if workers <= 1:
        for a in jobs:
            try:
                print(fn(*a), flush=True)
            except Exception:
                print(f"[FAIL] {a[0]}\n{traceback.format_exc()}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_safe, fn, a): a for a in jobs}
        for f in as_completed(futs):
            print(f.result(), flush=True)


def _safe(fn, a):
    import traceback
    try:
        return fn(*a)
    except Exception:
        return f"[FAIL] {a[0]}\n{traceback.format_exc()}"


def common_args(desc: str, default_groups=("dev",)):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="dataset names (default: all of the group)")
    ap.add_argument("--groups", nargs="*", default=list(default_groups), help="dev, val11, val2")
    return ap


def jobs_for(groups, only=None):
    out = []
    for g in groups:
        names = dev_names() if g == "dev" else val11_names() if g == "val11" else val2_names()
        out += [(n, g) for n in names if only is None or n in only]

    def size(j):
        X, _ = load_any(*j)
        return -X.size
    return sorted(out, key=size)


def val2_names():
    p = VAL2_DIR / "MANIFEST.json"
    if not p.exists():
        return []
    man = json.loads(p.read_text())
    return [n for n, m in man.get("datasets", {}).items() if "dropped" not in m and (VAL2_DIR / f"{n}.npz").exists()]
