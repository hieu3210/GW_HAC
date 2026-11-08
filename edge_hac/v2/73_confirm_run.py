"""Confirmatory runs on the pre-registered OpenML datasets (data/validation2), exactly as in PREREG_E7_FILLED.md.

Refuses to run unless SELECTION_LOCK.json records an OSF id and the pre-registration file is unchanged.
Methods: GW-HAC (HA-grid | SQM | interval, level 3, uniform weights), FL-U8, k-means, GW-nw (min-max preprocessing);
secondary: GW-HAC, FL-U8, k-means with log1p preprocessing of skewed non-negative features.
Theory outputs: decomposition on the k-means partition (HA-grid, levels 2-5) for P1; grid-calibrated alpha at
levels 2 and 5 for P2.

Output: <out>/raw/v2conf_<ds>.csv, v2confdec_<ds>.csv, v2confalpha_<ds>.csv

    python v2/73_confirm_run.py --workers 8
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

import v2lib as L
from edgehac import datasets as D


def check_lock() -> None:
    lock = json.loads((L.VAL2_DIR / "SELECTION_LOCK.json").read_text())
    if not lock.get("osf_id"):
        raise SystemExit("STOP: no OSF id recorded; run 72_fetch_validation.py --osf-id ... first.")
    if hashlib.sha256((L.REPO / lock["prereg_file"]).read_bytes()).hexdigest() != lock["prereg_sha256"]:
        raise SystemExit("STOP: pre-registration file changed.")


def run_dataset(name: str, group: str, out: Path) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / f"v2confalpha_{name}.csv").exists():
        return f"[skip] {name}"
    t0 = time.time()
    X, y = L.load_any(name, group)
    K = int(len(np.unique(y)))
    rows, dec, alp = [], [], []
    for sp in range(L.n_splits_for(*X.shape)):
        tr, te = D.split(len(X), sp)
        S_tr, S_te, _ = L.prepare_split(X, tr, te)
        F = S_tr.shape[1]
        if F == 0:
            return f"[FAIL] {name}: no non-constant feature"
        w = L.uniform_w(F)

        def add(method, lab):
            rows.append(dict(dataset=name, split=sp, method=method, ari=float(ARI(y[te], lab)), F=F, K=K))

        cache = {}
        voc = L.family_vocab("HA-grid", L.MAIN_LEVEL, S_tr, cache)
        f, hits, *_, lab = L.word_model(S_tr, S_te, voc, K, sp, w)
        add("GW-HAC", lab)
        rows[-1]["cap_hits"] = hits
        add("FL-U8", L.fl_uniform(S_tr, S_te, K, L.CFG["fl_labels"], sp)[-1])
        add("KM", L.kmeans(S_tr, S_te, K, sp)[-1])
        add("GW-nw", L.gw_nw(S_tr, S_te, K, sp, w))
        # registered secondary analysis: log1p preprocessing (descriptive, no decision)
        L_tr, L_te, _ = L.prepare_split(X, tr, te, "log1p")
        wl = L.uniform_w(L_tr.shape[1])
        vl = L.family_vocab("HA-grid", L.MAIN_LEVEL, L_tr, {})
        add("GW-HAC|log1p", L.word_model(L_tr, L_te, vl, K, sp, wl)[-1])
        add("FL-U8|log1p", L.fl_uniform(L_tr, L_te, K, L.CFG["fl_labels"], sp)[-1])
        add("KM|log1p", L.kmeans(L_tr, L_te, K, sp)[-1])
        km = KMeans(K, n_init=L.CFG["kmeans_n_init"], random_state=sp).fit(S_tr)
        for lvl in L.LEVELS:
            d = L.decomposition(S_tr, km.labels_, L.family_vocab("HA-grid", lvl, S_tr, cache), w)
            dec.append(dict(dataset=name, split=sp, family="HA-grid", level=lvl, **d))
        band = L.CFG["alpha_band"]
        for lvl in (2, 5):
            _, al = L.calibrate_grid(S_tr, lvl)
            alp.append(dict(dataset=name, split=sp, level=lvl, share_in_band=float(np.mean((al >= band[0]) & (al <= band[1]))),
                            F=F))
    pd.DataFrame(rows).to_csv(raw / f"v2conf_{name}.csv", index=False)
    pd.DataFrame(dec).to_csv(raw / f"v2confdec_{name}.csv", index=False)
    pd.DataFrame(alp).to_csv(raw / f"v2confalpha_{name}.csv", index=False)
    return f"[done] {name} N={len(X)} F={X.shape[1]} K={K} in {time.time() - t0:.0f}s"


def main() -> None:
    ap = L.common_args(__doc__, default_groups=("val2",))
    ap.add_argument("--no-lock-check", action="store_true", help="testing only; never for the confirmatory run")
    a = ap.parse_args()
    if not a.no_lock_check:
        check_lock()
    L.run_pool(run_dataset, [(n, g, Path(a.out)) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
