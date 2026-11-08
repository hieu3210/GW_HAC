"""E4 (v2): objective interpretability proxies (no user study).

  fidelity   - post hoc labelling: a numeric model (k-means, uniform weights) or FL-U8 is fitted first, its prototypes
               are then named (HA-grid words / winning FL label), and held-out samples are re-assigned from the names
               alone.  Agreement with the model's own assignment at levels 3, 2, 1 (HA words projected by dropping
               hedges).  GW-HAC rows give the agreement of its projected descriptions with its own decision.
  consistency- refit at the coarse level versus projection of the fine model (sample share and Hungarian-matched
               prototype share; definitions of 27_gw_interval.consistency), families HA-grid, HA-theta, dyadic,
               Quantile, level pairs (2,3), (3,4), (4,5).
  desclen    - distinguishing words per cluster (prototype word differs from the feature's modal word) at levels 3
               and 2, features needed for 90 % fidelity (H6 rule of the study), and IMM / ExKMC depth if installed.

Output: <out>/raw/v2int_<ds>.csv  (long format: kind, method, level, value)

    python v2/62_dev_interp.py --workers 8
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import v2lib as L
from edgehac import datasets as D
from edgehac.whac import distinguishing_words, embed, predict_words, semantize_words


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / f"v2int_{name}.csv"
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
        fidx = np.arange(F)[None, :]
        cache = {}

        def add(kind, method, level, value):
            rows.append(dict(dataset=name, group=group, split=sp, kind=kind, method=method, level=level,
                             value=float(value)))

        voc = {l: L.family_vocab("HA-grid", l, S_tr, cache) for l in (1, 2, 3, 4, 5)}
        # ---- fidelity of post hoc naming ------------------------------------------------------------------
        km, g_num = L.kmeans(S_tr, S_te, K, sp)
        Pz = semantize_words(km.cluster_centers_, voc[3])          # name the numeric prototypes
        qz = semantize_words(S_te, voc[3])
        for m in (0, 1, 2):
            g = predict_words(qz >> m, voc[3 - m], Pz >> m, weights=w)
            add("fidelity", "KM+HA-words", 3 - m, np.mean(g == g_num))
        f, _, q_tr, q_te, g_gw = L.word_model(S_tr, S_te, voc[3], K, sp, w)
        for m in (0, 1, 2):
            g = predict_words(q_te >> m, voc[3 - m], f.P >> m, weights=w)
            add("fidelity", "GW-HAC", 3 - m, np.mean(g == g_gw))
        T = L.CFG["fl_labels"]
        flkm, C, g_fl = L.fl_uniform(S_tr, S_te, K, T, sp)
        words = flkm.cluster_centers_.reshape(K, F, T).argmax(2)    # decoded label of each prototype
        cval = np.linspace(0, 1, T)
        g = L.wdist(L.fl_win_values(S_te, T), cval[words], w).argmin(1)
        add("fidelity", "FL-U8", 3, np.mean(g == g_fl))
        # ---- consistency: refit versus projection ----------------------------------------------------------
        for fam in ("HA-grid", "HA-theta", "dyadic", "Quantile"):
            fits = {}
            for l in (2, 3, 4, 5):
                vv = L.family_vocab(fam, l, S_tr, cache)
                ff, _, _, qq, _ = L.word_model(S_tr, S_te, vv, K, sp, w)
                fits[l] = (vv, ff, qq)
            for l in (2, 3, 4):
                vc, fc, qc = fits[l]
                vf, ff, qf = fits[l + 1]
                pm = (np.tile(np.arange(2 ** (l + 1)) // 2, (F, 1)) if fam == "Quantile"
                      else np.stack([[vc.names[j].index(wd[1:]) for wd in vf.names[j]] for j in range(F)]))
                samp, proto = L.GWI.consistency(vc, fc, qc, vf, ff, qf, w, pm)
                add("consistency_sample", fam, l, samp)
                add("consistency_proto", fam, l, proto)
        # ---- description length ------------------------------------------------------------------------------
        for m in (0, 1):
            dm = distinguishing_words(f.P >> m, q_tr >> m)
            add("desc_words_per_cluster", "GW-HAC", 3 - m, dm.sum(1).mean())
        nuP = voc[3].V[fidx, f.P]
        nuX = embed(q_te, voc[3])
        order = np.argsort(-(w * nuP.var(0)), kind="stable")
        need = F
        for mm in range(1, F + 1):
            sel = order[:mm]
            if np.mean(L.wdist(nuX[:, sel], nuP[:, sel], w[sel]).argmin(1) == g_gw) >= 0.9:
                need = mm
                break
        add("features_for_90pct", "GW-HAC", 3, need)
        add("features_total", "GW-HAC", 3, F)
        for kind, (lab, tree) in L.threshold_trees(S_tr, S_te, K, sp).items():
            add("tree_depth", kind, 0, L.tree_depth(tree, S_te))
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} ({group}) in {time.time() - t0:.0f}s"


def main() -> None:
    a = L.common_args(__doc__).parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
