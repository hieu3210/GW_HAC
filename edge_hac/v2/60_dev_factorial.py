"""E1 + E2 (v2): factorial design (family x representative x assignment x level) and matched baselines.

Uniform feature weights.  Families: dyadic, HA-theta (closed-form Medium, alpha = 1/2), HA-grid (19 x 19 at level 4,
the published GW-HAC calibration), Beta-warp, nested quantile.  Representatives: native (SQM for HA), cell midpoint,
training cell mean.  Assignment: interval cells or nearest representative (HA families).  Levels 2-5.
Baselines: k-means, FL-U8 (primary comparator), FL-U15 (legacy), FL-Q8, FL-U8-win (F-dimensional), GW-nw, IMM/ExKMC.

Output: <out>/raw/v2fac_<ds>.csv   (one row per dataset x split x method x level)

    python v2/60_dev_factorial.py --workers 8                 # 25 development datasets
    python v2/60_dev_factorial.py --groups val11 --workers 8  # exploratory: the 11 first-validation datasets
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

import v2lib as L
from edgehac import datasets as D
from edgehac.baselines_gw import memberships, ruspini_centres


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / f"v2fac_{name}.csv"
    if dest.exists():
        return f"[skip] {name}"
    t0 = time.time()
    X, y = L.load_any(name, group)
    K = int(len(np.unique(y)))
    rows = []
    for sp in range(L.n_splits_for(*X.shape)):
        tr, te = D.split(len(X), sp)
        S_tr, S_te, _ = L.prepare_split(X, tr, te)
        F = S_tr.shape[1]
        w = L.uniform_w(F)
        y_te = y[te]

        def add(method, level, lab, **extra):
            rows.append(dict(dataset=name, group=group, split=sp, method=method, level=level, F=F, K=K,
                             ari=float(ARI(y_te, lab)), nmi=float(NMI(y_te, lab)), **extra))

        cache = {}
        for cfg in L.FACTORIAL:
            fam, rep, rule = cfg
            for lvl in L.LEVELS:
                voc = L.with_assign(L.with_rep(L.family_vocab(fam, lvl, S_tr, cache), rep, S_tr), rule)
                f, hits, q_tr, q_te, lab = L.word_model(S_tr, S_te, voc, K, sp, w)
                add(L.tag(cfg), lvl, lab, family=fam, rep=rep, rule=rule, J=f.J, n_iter=f.n_iter, cap_hits=hits,
                    mse_te=float(L.vocab_mse(S_te, voc).mean()), mse_tr=float(L.vocab_mse(S_tr, voc).mean()))
        # ---- baselines --------------------------------------------------------------------------------------
        _, lab = L.kmeans(S_tr, S_te, K, sp)
        add("KM", 0, lab)
        for T in (L.CFG["fl_labels"], L.CFG["fl_labels_legacy"]):
            _, _, lab = L.fl_uniform(S_tr, S_te, K, T, sp)
            add(f"FL-U{T}", 0, lab)
        C = ruspini_centres(L.CFG["fl_labels"], S_tr)
        km = KMeans(K, n_init=L.CFG["kmeans_n_init"], random_state=sp).fit(memberships(S_tr, C).reshape(len(S_tr), -1))
        add(f"FL-Q{L.CFG['fl_labels']}", 0, km.predict(memberships(S_te, C).reshape(len(S_te), -1)))
        add(f"FL-U{L.CFG['fl_labels']}-win", 0, L.fl_win(S_tr, S_te, K, L.CFG["fl_labels"], sp))
        add("GW-nw", 0, L.gw_nw(S_tr, S_te, K, sp, w))
        for kind, (lab, tree) in L.threshold_trees(S_tr, S_te, K, sp).items():
            add(kind, 0, lab, depth=L.tree_depth(tree, S_te))
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} ({group}) N={len(X)} F={X.shape[1]} K={K} in {time.time() - t0:.0f}s"


def main() -> None:
    a = L.common_args(__doc__).parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
