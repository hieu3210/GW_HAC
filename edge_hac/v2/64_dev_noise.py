"""E6 (v2): robustness of the held-out assignment to measurement noise (secondary, descriptive).

Gaussian noise N(0, sigma^2) is added to the held-out semantic coordinates (then clipped to [0, 1]); the share of
samples whose cluster changes is recorded.  Methods: GW-HAC (HA-grid, level 3), dyadic level 3, k-means, FL-U8.

Output: <out>/raw/v2noi_<ds>.csv

    python v2/64_dev_noise.py --workers 8
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import v2lib as L
from edgehac import datasets as D
from edgehac.baselines_gw import memberships
from edgehac.whac import predict_words, semantize_words


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / f"v2noi_{name}.csv"
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
        cache = {}
        pred = {}
        for fam, tagname in (("HA-grid", "GW-HAC"), ("dyadic", "dyadic")):
            voc = L.family_vocab(fam, L.MAIN_LEVEL, S_tr, cache)
            f = L.word_model(S_tr, S_te, voc, K, sp, w)[0]
            pred[tagname] = (lambda S, voc=voc, P=f.P: predict_words(semantize_words(S, voc), voc, P, weights=w))
        km, _ = L.kmeans(S_tr, S_te, K, sp)
        pred["KM"] = km.predict
        flkm, C, _ = L.fl_uniform(S_tr, S_te, K, L.CFG["fl_labels"], sp)
        pred["FL-U8"] = lambda S, flkm=flkm, C=C: flkm.predict(memberships(S, C).reshape(len(S), -1))
        clean = {k: p(S_te) for k, p in pred.items()}
        for sigma in L.CFG["noise_sigmas"]:
            for ns in L.CFG["noise_seeds"]:
                rng = np.random.default_rng(10_000 * sp + ns)
                Sn = np.clip(S_te + rng.normal(0, sigma, S_te.shape), 0, 1)
                for k, p in pred.items():
                    rows.append(dict(dataset=name, group=group, split=sp, method=k, sigma=sigma, noise_seed=ns,
                                     flip=float(np.mean(p(Sn) != clean[k]))))
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} ({group}) in {time.time() - t0:.0f}s"


def main() -> None:
    a = L.common_args(__doc__).parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
