"""E5 (v2): skewed / zero-inflated features (the Spambase case) and three preprocessing rules.

Preprocessing (training half only): minmax (default), log1p for non-negative features with skew > 2, training ECDF
("rank").  Methods: GW-HAC (HA-grid, level 3), HA-theta (level 3), FL-U8, k-means.  Diagnostics per split:
share of features whose most frequent level-3 cell holds >= 90 % of the training values (minmax, HA-grid),
mean |skew|, share of features with >= 30 % of the values at their minimum.

Output: <out>/raw/v2skw_<ds>.csv

    python v2/63_dev_skew.py --groups dev val11 --workers 8
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import adjusted_rand_score as ARI

import v2lib as L
from edgehac import datasets as D
from edgehac.whac import semantize_words


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / f"v2skw_{name}.csv"
    if dest.exists():
        return f"[skip] {name}"
    t0 = time.time()
    X, y = L.load_any(name, group)
    K = int(len(np.unique(y)))
    rows = []
    for sp in range(L.n_splits_for(*X.shape)):
        tr, te = D.split(len(X), sp)
        for mode in ("minmax", "log1p", "rank"):
            S_tr, S_te, _ = L.prepare_split(X, tr, te, mode)
            F = S_tr.shape[1]
            w = L.uniform_w(F)
            cache = {}
            res = {}
            for fam, tagname in (("HA-grid", "GW-HAC"), ("HA-theta", "HA-theta")):
                voc = L.family_vocab(fam, L.MAIN_LEVEL, S_tr, cache)
                res[tagname] = L.word_model(S_tr, S_te, voc, K, sp, w)[-1]
                if fam == "HA-grid":
                    q = semantize_words(S_tr, voc)
                    top = np.array([np.bincount(q[:, j], minlength=voc.T).max() / len(q) for j in range(F)])
            res["FL-U8"] = L.fl_uniform(S_tr, S_te, K, L.CFG["fl_labels"], sp)[-1]
            res["KM"] = L.kmeans(S_tr, S_te, K, sp)[-1]
            sk = skew(S_tr, axis=0)
            atmin = (S_tr <= S_tr.min(0) + 1e-12).mean(0)
            for meth, lab in res.items():
                rows.append(dict(dataset=name, group=group, split=sp, prep=mode, method=meth, ari=float(ARI(y[te], lab)),
                                 share_top90=float(np.mean(top >= 0.9)), mean_abs_skew=float(np.nanmean(np.abs(sk))),
                                 share_atmin30=float(np.mean(atmin >= 0.3)), F=F, K=K))
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} ({group}) in {time.time() - t0:.0f}s"


def main() -> None:
    a = L.common_args(__doc__, default_groups=("dev", "val11")).parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
