"""Wall-clock timing of GW-HAC and the comparators (manuscript v2.5, Section V-B "Computational cost").

Times, on every split of every dataset, the stages used in the paper (single thread, sequential):
  calib_grid   HA-grid calibration (19 x 19 grid, training distortion at level 4)
  calib_theta  closed-form Medium (HA-theta)
  semantize    level-3 word assignment of training and held-out data
  fit_gwhac    GW-HAC fit, best of 5 (Algorithm 1), level 3
  FL-U8, KM, GW-nw   comparators, fit + predict, as in 73_confirm_run.py
  IMM+ExKMC    threshold trees (only if ExKMC is installed; includes their k-means fit)

Run with ONE worker, so that the timings do not compete for cores:
    python v2/75_timing.py --workers 1 --groups dev val2
Output: <out>/raw/v2time_<ds>.csv   (dataset, group, split, N, F, K, stage, seconds)
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import v2lib as L
from edgehac import datasets as D
from edgehac.whac import semantize_words


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / f"v2time_{name}.csv"
    if dest.exists():
        return f"[skip] {name}"
    X, y = L.load_any(name, group)
    K = int(len(np.unique(y)))
    rows = []
    for sp in range(L.n_splits_for(*X.shape)):
        tr, te = D.split(len(X), sp)
        S_tr, S_te, _ = L.prepare_split(X, tr, te)
        F = S_tr.shape[1]
        if F == 0:
            break
        w = L.uniform_w(F)

        def rec(stage, fn):
            t0 = time.perf_counter()
            r = fn()
            rows.append(dict(dataset=name, group=group, split=sp, N=len(X), F=F, K=K, stage=stage,
                             seconds=time.perf_counter() - t0))
            return r

        cache = {}
        rec("calib_grid", lambda: L.family_vocab("HA-grid", L.MAIN_LEVEL, S_tr, cache))
        rec("calib_theta", lambda: L.theta_hat(S_tr))
        voc = L.family_vocab("HA-grid", L.MAIN_LEVEL, S_tr, cache)
        q_tr, q_te = rec("semantize", lambda: (semantize_words(S_tr, voc), semantize_words(S_te, voc)))
        rec("fit_gwhac", lambda: L.best_of_v2(q_tr, voc, K, sp, w))
        rec("FL-U8", lambda: L.fl_uniform(S_tr, S_te, K, L.CFG["fl_labels"], sp))
        rec("KM", lambda: L.kmeans(S_tr, S_te, K, sp))
        rec("GW-nw", lambda: L.gw_nw(S_tr, S_te, K, sp, w))
        try:
            import ExKMC  # noqa: F401
            rec("IMM+ExKMC", lambda: L.threshold_trees(S_tr, S_te, K, sp))
        except Exception:
            pass
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[done] {name} ({group}) N={len(X)} F={X.shape[1]} K={K}"


def main() -> None:
    a = L.common_args(__doc__, default_groups=("dev", "val2")).parse_args()
    if a.workers != 1:
        print("[warn] timings are only comparable with --workers 1")
    (Path(a.out) / "logs").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "logs" / "75_machine.txt").write_text(
        f"{platform.platform()}\n{platform.processor()}\npython {platform.python_version()}\nnumpy {np.__version__}\n")
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
