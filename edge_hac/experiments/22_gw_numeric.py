"""GW-HAC phase C, numeric datasets (groups A + C): H1 (without TSK), H3, H4, H5, H6 and S1.

Protocol: docs/unified/PREREG_UNIFIED.md (frozen) + ADDENDUM_U1.md; configs/protocol_unified.json.
Outputs (restart-safe) in <out>/raw: num_<ds>.csv (scores), h3_<ds>.csv, h4_<ds>.csv, h5_<ds>.csv, h6_<ds>.csv;
h6_<ds>.csv is the completion marker.

    python experiments/22_gw_numeric.py --out ../results/GW-HAC_2026.10.27 --workers 8
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
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

from edgehac import datasets as D
from edgehac import sqm
from edgehac.baselines import kmeans_float, nearest_centroid
from edgehac.baselines_gw import memberships, ruspini_centres
from edgehac.edge_model import compile_edge
from edgehac.fit import FittedHA, fit_ewahac, float_predict, semantize
from edgehac.infer import run, run_indices, to_codes
from edgehac.lattice import ha_lattice
from edgehac.whac import (bins_equal_width, best_of, embed, fit_words, kmodes, kmodes_predict, make_vocab,
                          predict_words, quantile_vocab, semantize_words, uniform_vocab)

PROTO = json.loads((HERE / "configs" / "protocol_unified.json").read_text())
SPECS = PROTO["semantics"]["specificities"]
N_INIT = PROTO["GW"]["n_init"]
GRID = np.linspace(*PROTO["H5"]["grid_range"], PROTO["H5"]["grid"])
N_LAB = PROTO["H5"]["n_labels"]


def n_splits_for(N: int, F: int) -> int:
    P = PROTO["protocol_numeric"]
    return P["splits_if_NF_gt_1e6"] if N * F > 1_000_000 else P["splits"]


def wdist(A, C, w, chunk=2048):
    out = np.empty((len(A), len(C)))
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = np.sqrt(np.einsum("nkf,f->nk", (A[i:i + chunk, None, :] - C[None]) ** 2, w))
    return out


def nearest_idx(S, v):
    return np.searchsorted(0.5 * (v[1:] + v[:-1]), S, side="left")


def scores(y, lab):
    return dict(ari=float(ARI(y, lab)), nmi=float(NMI(y, lab)))


# ---------------------------------------------------------------------------------------------------------
def t1_checks(S, C, w, v):
    """T1.1 / T1.2 on held-out data: N prototypes and data translated into the words of the shared vocabulary v."""
    N, K = len(S), len(C)
    r = sqm.covering_radius(v)
    dN = wdist(S, C, w)
    l = dN.argmin(1)
    JN = float((dN[np.arange(N), l] ** 2).sum())
    srt = np.sort(dN, 1)
    margin = srt[:, 1] - srt[:, 0] if K > 1 else np.full(N, np.inf)
    Xw, Pw = v[nearest_idx(S, v)], v[nearest_idx(C, v)]
    dW = wdist(Xw, Pw, w)
    Jtr = float((dW[np.arange(N), l] ** 2).sum())
    E, Dl = Xw - S, Pw - C
    dev = np.sqrt(np.einsum("nkf,f->nk", (E[:, None, :] - Dl[None]) ** 2, w))
    other = np.max(np.where(np.arange(K)[None] == l[:, None], -np.inf, dev), 1)
    safe = margin > dev[np.arange(N), l] + other
    chg = dW.argmin(1) != l
    bound = 2 * r * np.sqrt(N * w.sum())
    gap = abs(np.sqrt(Jtr) - np.sqrt(JN))
    return dict(r=r, JN=JN, Jtr=Jtr, bound=bound, viol_T11=bool(gap > bound + 1e-9),
                nontrivial_T11=bool(bound < np.sqrt(JN)), cert_T12=float(safe.mean()),
                viol_T12=int((safe & chg).sum()), chg=float(chg.mean()), viol_T12_static=int(((margin > 4 * r) & chg).sum()))


def word_exact(voc, P, q_te, w):
    """Integer realizations of the weighted word rule against its real-valued evaluation (exact word inputs)."""
    F = q_te.shape[1]
    nuP = voc.V[np.arange(F)[None, :], P]
    ref = predict_words(q_te, voc, P, weights=w)
    fit = FittedHA(centers=nuP, weights=w, lo=np.zeros(F), hi=np.ones(F), theta=.5, alpha=.5, gamma=1.0, n_iter=1,
                   train_labels=np.zeros(1, int))
    out = {}
    for name, kw in (("lut12", dict(realization="lut", lut_bits=12)), ("mac16", dict(realization="mac", weight_bits=16))):
        m = compile_edge(fit, voc.lattices(), B=12, exact_inputs=True, **kw)
        res = run_indices(m, q_te)
        wrong = res.labels != ref
        out[name] = dict(fidelity=float(1 - wrong.mean()), cov_pairwise=float(res.cert_pairwise.mean()),
                         cov_global=float(res.cert_global.mean()), wrong_cert_p=int((res.cert_pairwise & wrong).sum()),
                         wrong_cert_g=int((res.cert_global & wrong).sum()))
    return out


def tree_depths(tree, X):
    d = np.zeros(len(X))
    for i, x in enumerate(X):
        node, k = tree.tree, 0
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.value else node.right
            k += 1
        d[i] = k
    return d


def fl_fit_predict(S_tr, S_te, K, T, method, placement, seed):
    C = ruspini_centres(T) if placement == "uniform" else ruspini_centres(T, S_tr)
    Z_tr, Z_te = memberships(S_tr, C).reshape(len(S_tr), -1), memberships(S_te, C).reshape(len(S_te), -1)
    if method == "kmeans":
        km = KMeans(K, n_init=PROTO["FL"]["kmeans_n_init"], random_state=seed).fit(Z_tr)
        return km.predict(Z_te)
    import skfuzzy
    cntr, *_ = skfuzzy.cluster.cmeans(Z_tr.T, K, PROTO["FL"]["fcm_m"], error=1e-5, maxiter=300, seed=seed)
    u, *_ = skfuzzy.cluster.cmeans_predict(Z_te.T, cntr, PROTO["FL"]["fcm_m"], error=1e-5, maxiter=300, seed=seed)
    return u.argmax(0)


def labelled_subset(y, n, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    first = [rng.choice(np.flatnonzero(y == c)) for c in classes] if n >= len(classes) else []
    rest = rng.permutation(np.setdiff1d(np.arange(len(y)), first))
    return np.array(list(first) + list(rest[:max(0, n - len(first))]), int)


# ---------------------------------------------------------------------------------------------------------
def run_dataset(name: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / f"h6_{name}.csv").exists():
        return f"[skip] {name}"
    t0 = time.time()
    data = D.load(name)
    X, y = data["X"], data["y"]
    K = int(len(np.unique(y)))
    N, F = X.shape
    rows, h3, h4, h5, h6 = [], [], [], [], []
    for sp in range(n_splits_for(N, F)):
        tr, te = D.split(N, sp)
        fit = fit_ewahac(X[tr], K, seed=sp)
        S_tr, S_te = semantize(X[tr], fit.lo, fit.hi), semantize(X[te], fit.lo, fit.hi)
        y_tr, y_te = y[tr], y[te]
        w = fit.weights / fit.weights.sum()
        ref_te, _, _ = float_predict(S_te, fit.centers, fit.weights)

        def add(method, s, lab, **extra):
            rows.append(dict(dataset=name, split=sp, method=method, spec=s, F=F, K=K, **scores(y_te, lab), **extra))

        add("N", 0, ref_te)
        kmC, _ = kmeans_float(S_tr, K, seed=sp)
        add("KM", 0, nearest_centroid(S_te, kmC)[0])
        # ---- H3 for N: integer LUT realization with the completed Little lattice --------------------------
        lat = ha_lattice(.5, .5, depth=2, completed=True)
        m = compile_edge(fit, [lat] * F, B=12, realization="lut", lut_bits=8)
        res = run(m, to_codes(S_te, 12))
        wrong = res.labels != ref_te
        h3.append(dict(dataset=name, split=sp, model="N", real="lut8", fidelity=float(1 - wrong.mean()),
                       cov_pairwise=float(res.cert_pairwise.mean()), cov_global=float(res.cert_global.mean()),
                       wrong_cert_p=int((res.cert_pairwise & wrong).sum()), wrong_cert_g=int((res.cert_global & wrong).sum())))
        # ---- threshold trees (authors' library) ----------------------------------------------------------
        from ExKMC.Tree import Tree
        km = KMeans(K, n_init=10, random_state=sp).fit(S_tr)
        trees = {}
        for kind, leaves in (("IMM", K), ("ExKMC", 2 * K)):
            t = Tree(k=K, max_leaves=leaves, base_tree="IMM", random_state=sp)
            t.fit(S_tr, km)
            trees[kind] = t
            add(kind, 0, t.predict(S_te), depth=float(tree_depths(t, S_te).mean()))
        # ---- word models per specificity ------------------------------------------------------------------
        gw, uw = {}, {}
        for s in SPECS:
            T = 2 ** (s + 2) - 1
            for variant in (["little", "rather"] if s == PROTO["semantics"]["primary_spec"] else ["little"]):
                voc = make_vocab(.5, .5, F, s, variant=variant)
                q_tr, q_te = semantize_words(S_tr, voc), semantize_words(S_te, voc)
                f = best_of(q_tr, voc, K, n_init=N_INIT, seed=sp, weights=w)
                add("GW" if variant == "little" else "GW-rather", s, predict_words(q_te, voc, f.P, weights=w), J=f.J)
                if variant == "little":
                    gw[s] = (voc, f, q_tr, q_te)
                    for real, mm in word_exact(voc, f.P, q_te, w).items():
                        h3.append(dict(dataset=name, split=sp, model=f"GW-s{s}", real=real, **mm))
                    h3.append(dict(dataset=name, split=sp, model=f"T1-s{s}", real="",
                                   **t1_checks(S_te, fit.centers, w, voc.V[0])))
            for meth, voc in (("UW", uniform_vocab(F, T)), ("QW", quantile_vocab(S_tr, T))):
                q_tr, q_te = semantize_words(S_tr, voc), semantize_words(S_te, voc)
                f = best_of(q_tr, voc, K, n_init=N_INIT, seed=sp, weights=w)
                add(meth, s, predict_words(q_te, voc, f.P, weights=w))
                if meth == "UW":
                    uw[s] = (voc, f, q_tr, q_te)
            B_tr, B_te = bins_equal_width(S_tr, T), bins_equal_width(S_te, T)
            modes, _, _ = kmodes(B_tr, K, T, n_init=5, seed=sp)
            add("KMO", s, kmodes_predict(B_te, modes))
            for meth, (mt, pl) in (("FL-U", ("kmeans", "uniform")), ("FL-Q", ("kmeans", "quantile")),
                                   ("FCM-U", ("fcm", "uniform"))):
                add(meth, s, fl_fit_predict(S_tr, S_te, K, T, mt, pl, sp), n_params_per_feature=3 * T)
        # ---- H4: granularity consistency (s -> s+1) ----------------------------------------------------------
        for s in SPECS[:-1]:
            vc, fc, _, qc = gw[s]
            vf, ff, _, qf = gw[s + 1]
            names_f, names_c = vf.names[0], vc.names[0]
            proj = {i: names_c.index(wd[1:] if len(wd) - 1 == s + 1 else wd) for i, wd in enumerate(names_f)}
            pmap = np.array([proj[i] for i in range(len(names_f))])
            samp_gw = float(np.mean(pmap[qf] == qc))
            nu_c = vc.V[0][fc.P]
            nu_f_proj = vc.V[0][pmap[ff.P]]
            C = np.sqrt(np.einsum("abf,f->ab", (nu_c[:, None] - nu_f_proj[None]) ** 2, w))
            r_, c_ = linear_sum_assignment(C)
            proto_gw = float(np.mean(fc.P[r_] == pmap[ff.P][c_]))
            uvc, ufc, _, uqc = uw[s]
            uvf, uff, _, uqf = uw[s + 1]
            umap = nearest_idx(uvf.V[0], uvc.V[0])
            samp_uw = float(np.mean(umap[uqf] == uqc))
            nu_uc, nu_uf = uvc.V[0][ufc.P], uvc.V[0][umap[uff.P]]
            Cu = np.sqrt(np.einsum("abf,f->ab", (nu_uc[:, None] - nu_uf[None]) ** 2, w))
            ru, cu = linear_sum_assignment(Cu)
            proto_uw = float(np.mean(ufc.P[ru] == umap[uff.P][cu]))
            h4.append(dict(dataset=name, split=sp, s=s, sample_GW=samp_gw, sample_UW=samp_uw, proto_GW=proto_gw,
                           proto_UW=proto_uw))
        # ---- H5: (a) distortion, (b) few labels, (c) shape of marginals, (d) alpha* ----------------------------
        s2 = PROTO["semantics"]["primary_spec"]
        vocs = {(th, al): sqm.vocabulary(th, al, s2, "little")[0] for th in GRID for al in GRID}
        mse_tr = {k: ((v[nearest_idx(S_tr, v)] - S_tr) ** 2).mean(0) for k, v in vocs.items()}
        keys = list(vocs)
        best = [keys[int(np.argmin([mse_tr[k][j] for k in keys]))] for j in range(F)]
        mse_te_cal = float(np.mean([((vocs[best[j]][nearest_idx(S_te[:, j], vocs[best[j]])] - S_te[:, j]) ** 2).mean()
                                    for j in range(F)]))
        v_half = sqm.vocabulary(.5, .5, s2, "little")[0]
        mse_te_half = float(((v_half[nearest_idx(S_te, v_half)] - S_te) ** 2).mean())
        astar = 1.0 / (1.0 + 2.0 ** (-1.0 / (s2 + 1)))
        v_star = sqm.vocabulary(.5, astar, s2, "little")[0]
        mse_te_star = float(((v_star[nearest_idx(S_te, v_star)] - S_te) ** 2).mean())
        grid_labels = {}
        for k, v in vocs.items():
            voc = make_vocab(k[0], k[1], F, s2, variant="little")
            grid_labels[k] = fit_words(semantize_words(S_tr, voc), voc, K, weights=w).labels
        half_voc = gw[s2][0]
        half_te = predict_words(gw[s2][3], half_voc, gw[s2][1].P, weights=w)
        rec = dict(dataset=name, split=sp, mse_te_half=mse_te_half, mse_te_cal=mse_te_cal, mse_te_astar=mse_te_star,
                   ari_half=float(ARI(y_te, half_te)), mean_abs_skew=float(np.mean(np.abs(skew(S_tr, axis=0, nan_policy="omit")))),
                   mean_ex_kurt=float(np.nanmean(kurtosis(S_tr, axis=0, nan_policy="omit"))))
        for n in N_LAB:
            idx = labelled_subset(y_tr, n, sp)
            sc = {k: ARI(y_tr[idx], lab[idx]) for k, lab in grid_labels.items()}
            top = max(sc.values())
            th, al = min([k for k, val in sc.items() if val == top], key=lambda k: (k[0] - .5) ** 2 + (k[1] - .5) ** 2)
            voc = make_vocab(th, al, F, s2, variant="little")
            f = best_of(semantize_words(S_tr, voc), voc, K, n_init=N_INIT, seed=sp, weights=w)
            rec[f"ari_lab{n}"] = float(ARI(y_te, predict_words(semantize_words(S_te, voc), voc, f.P, weights=w)))
            rec[f"theta_lab{n}"], rec[f"alpha_lab{n}"] = float(th), float(al)
        h5.append(rec)
        # ---- H6: description length at 90 % fidelity ----------------------------------------------------------
        voc, f, _, q_te = gw[s2]
        nuP = voc.V[np.arange(F)[None, :], f.P]
        nuX = embed(q_te, voc)
        full = predict_words(q_te, voc, f.P, weights=w)
        order = np.argsort(-(w * nuP.var(0)), kind="stable")
        m_req = F
        for m_ in range(1, F + 1):
            sel = order[:m_]
            lab = wdist(nuX[:, sel], nuP[:, sel], w[sel]).argmin(1)
            if np.mean(lab == full) >= PROTO["H6"]["fidelity"]:
                m_req = m_
                break
        h6.append(dict(dataset=name, split=sp, F=F, m_gw=m_req, depth_imm=float(tree_depths(trees["IMM"], S_te).mean()),
                       depth_exkmc=float(tree_depths(trees["ExKMC"], S_te).mean())))
    pd.DataFrame(rows).to_csv(raw / f"num_{name}.csv", index=False)
    pd.DataFrame(h3).to_csv(raw / f"h3_{name}.csv", index=False)
    pd.DataFrame(h4).to_csv(raw / f"h4_{name}.csv", index=False)
    pd.DataFrame(h5).to_csv(raw / f"h5_{name}.csv", index=False)
    pd.DataFrame(h6).to_csv(raw / f"h6_{name}.csv", index=False)
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
    print(f"{len(names)} datasets", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, (n, Path(a.out))) for n in order]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
