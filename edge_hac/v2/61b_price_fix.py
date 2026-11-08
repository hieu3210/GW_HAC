"""Correction of the price column of 61_dev_theory.py (Theorem 3 on real features).

Bug in v2lib.lloyd_1d (used by script 61): on features with tied values the quantile initialisation produced duplicate
centres, which ``np.unique`` removed permanently, so the "optimal" quantizer had fewer than 2^l levels and its
distortion was too large (prices < 1).  This script recomputes the reference distortion EXACTLY: the optimal 2^l-level
quantizer of the training values is found by dynamic programming over the sorted unique values (1-D k-means DP with
divide and conquer), so price >= 1 holds by construction.
The family codebooks are rebuilt from the (theta, alpha) stored in v2feat_<ds>.csv; nothing is re-calibrated.

Output: <out>/raw/v2price_<ds>.csv  (dataset, split, feature, level, mse_ref, price_<family>)

    python v2/61b_price_fix.py --workers 8
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import v2lib as L
from edgehac import datasets as D

FAMS = ("dyadic", "HA-theta", "HA-grid", "HA-grid-l")


def ref_mse(x, T, codebooks=()):
    return L.lloyd_1d(x, T)                         # exact DP (v2lib)


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    dest = raw / f"v2price_{name}.csv"
    if dest.exists():
        return f"[skip] {name}"
    t0 = time.time()
    feat = pd.read_csv(raw / f"v2feat_{name}.csv")
    X, y = L.load_any(name, group)
    rows = []
    for sp in sorted(feat.split.unique()):
        tr, te = D.split(len(X), int(sp))
        S_tr, _, keep = L.prepare_split(X, tr, te)
        cols = list(np.flatnonzero(keep))
        fs = feat[feat.split == sp]
        for _, r in fs.iterrows():
            j = cols.index(int(r.feature))
            lvl = int(r.level)
            x = S_tr[:, j]
            pars = {"dyadic": (.5, .5), "HA-theta": (r.theta_hat, .5), "HA-grid": (r.theta_grid4, r.alpha_grid4),
                    "HA-grid-l": (r.theta_grid_l, r.alpha_grid_l)}
            vocs = {k: L.GWI.level_cells(float(t), float(a), lvl) for k, (t, a) in pars.items()}
            mse_fam = {k: float(((v[np.searchsorted(b, x, side="right")] - x) ** 2).mean()) for k, (v, b, _) in vocs.items()}
            ref = ref_mse(x, 2 ** lvl, [v for v, _, _ in vocs.values()])
            row = dict(dataset=name, group=group, split=int(sp), feature=int(r.feature), level=lvl, mse_ref=ref,
                       n_unique=int(len(np.unique(x))))
            for k in FAMS:
                row[f"mse_{k}"] = mse_fam[k]
                row[f"price_{k}"] = mse_fam[k] / ref if ref > 0 else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} in {time.time() - t0:.0f}s"


def main() -> None:
    a = L.common_args(__doc__).parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
