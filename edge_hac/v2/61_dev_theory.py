"""E3 (v2): the theory of Section IV checked on real features, and the clustering results of Section V.

Per split (uniform weights):
  v2feat_<ds>.csv  per feature x level: median, theta_hat (Thm 2), grid (theta, alpha) calibrated AT that level,
                   alpha calibrated with theta = theta_hat, Q_hat (Prop 5), held-out / training distortion of
                   dyadic, HA-theta, HA-grid (level-4 calibration, as GW-HAC), and the optimal 2^l-point quantizer
                   of the training values (price, Thm 3).
  v2dec_<ds>.csv   Proposition 7 on the k-means partition (families dyadic, HA-theta, HA-grid; levels 2-5) and the
                   objective gap of the fitted GW-HAC partition.
  v2cert_<ds>.csv  Theorem 4 for k-means (uniform weights) and EWA-HAC (its own weights; --no-ewa to skip), both
                   conditions, all levels, and the agreement of GW-HAC with the numeric model.
  v2gran_<ds>.csv  Corollary 2 / Definition 2: certified granularity m*(i) of GW-HAC at levels 3, 4, 5.

    python v2/61_dev_theory.py --workers 8
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

import v2lib as L
from edgehac import datasets as D
from edgehac.fit import fit_ewahac, semantize

def run_dataset(name: str, group: str, out: Path, use_ewa: bool = True) -> str:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / f"v2gran_{name}.csv").exists():
        return f"[skip] {name}"
    t0 = time.time()
    X, y = L.load_any(name, group)
    K = int(len(np.unique(y)))
    feat, dec, cert, gran = [], [], [], []
    for sp in range(L.n_splits_for(*X.shape)):
        tr, te = D.split(len(X), sp)
        S_tr, S_te, keep = L.prepare_split(X, tr, te)
        F = S_tr.shape[1]
        w = L.uniform_w(F)
        cache = {}
        th_hat = L.theta_hat(S_tr)
        med = np.median(S_tr, 0)
        th_g, al_g = L.GWI.calibrate(S_tr)                         # GW-HAC calibration (level 4)
        cache["grid"], cache["theta_hat"] = (th_g, al_g), th_hat
        # ---- (a) per-feature theory -------------------------------------------------------------------------
        for lvl in L.LEVELS:
            th_l, al_l = L.calibrate_grid(S_tr, lvl)
            _, al_cond = L.calibrate_grid(S_tr, lvl, theta_fixed=th_hat)
            Q = L.Q_hat(S_tr, th_hat, lvl)
            mse = {}
            for fam in ("dyadic", "HA-theta", "HA-grid"):
                voc = L.family_vocab(fam, lvl, S_tr, cache)
                mse[fam] = (L.vocab_mse(S_tr, voc), L.vocab_mse(S_te, voc))
            voc_l = L.GWI.interval_vocab(th_l, al_l, lvl)
            mse["HA-grid-l"] = (L.vocab_mse(S_tr, voc_l), L.vocab_mse(S_te, voc_l))
            for j in range(F):
                lm = L.lloyd_1d(S_tr[:, j], 2 ** lvl)
                r = dict(dataset=name, group=group, split=sp, feature=int(np.flatnonzero(keep)[j]), level=lvl,
                         median=float(med[j]), theta_hat=float(th_hat[j]), theta_grid4=float(th_g[j]),
                         alpha_grid4=float(al_g[j]), theta_grid_l=float(th_l[j]), alpha_grid_l=float(al_l[j]),
                         alpha_given_theta_hat=float(al_cond[j]), Q_hat=float(Q[j]), mse_lloyd_tr=lm,
                         skew=float(pd.Series(S_tr[:, j]).skew()), top_cell_share_l3=np.nan)
                for k, (a_, b_) in mse.items():
                    r[f"mse_tr_{k}"], r[f"mse_te_{k}"] = float(a_[j]), float(b_[j])
                feat.append(r)
        voc3 = L.family_vocab("HA-grid", 3, S_tr, cache)
        q3 = L.semantize_words(S_tr, voc3)
        share = [np.bincount(q3[:, j], minlength=8).max() / len(q3) for j in range(F)]
        for r in feat[-F * len(L.LEVELS):]:
            r["top_cell_share_l3"] = float(share[list(np.flatnonzero(keep)).index(r["feature"])])
        # ---- (b) decomposition (Prop 7) and fitted-model gap ------------------------------------------------------
        km = KMeans(K, n_init=L.CFG["kmeans_n_init"], random_state=sp).fit(S_tr)
        JN_star = float((w * (S_tr - km.cluster_centers_[km.labels_]) ** 2).sum())
        for fam in ("dyadic", "HA-theta", "HA-grid"):
            for lvl in L.LEVELS:
                voc = L.family_vocab(fam, lvl, S_tr, cache)
                d = L.decomposition(S_tr, km.labels_, voc, w)
                f, hits, *_ = L.word_model(S_tr, S_te, voc, K, sp, w)
                lab = f.labels
                z = np.stack([S_tr[lab == k].mean(0) if np.any(lab == k) else np.zeros(F) for k in range(K)])
                JN_gw = float((w * (S_tr - z[lab]) ** 2).sum())
                dec.append(dict(dataset=name, group=group, split=sp, family=fam, level=lvl, N=len(S_tr), **d,
                                JN_star=JN_star, JN_of_gw_partition=JN_gw, J_word_gw=f.J,
                                gap_fit=(JN_gw - JN_star) / JN_star, gap_word=(f.J - JN_star) / JN_star))
        # ---- (c) translation certificates (Thm 4) ---------------------------------------------------------------
        models = [("KM", km.cluster_centers_, w, S_tr, S_te)]
        if use_ewa:
            fit = fit_ewahac(X[tr][:, keep], K, seed=sp)
            S_tr_e, S_te_e = semantize(X[tr][:, keep], fit.lo, fit.hi), semantize(X[te][:, keep], fit.lo, fit.hi)
            models.append(("EWA-HAC", fit.centers, fit.weights / fit.weights.sum(), S_tr_e, S_te_e))
        for mname, Z, wm, Str, Ste in models:
            cache_m = {} if mname != "KM" else cache
            for fam in ("dyadic", "HA-grid"):
                for lvl in L.LEVELS:
                    voc = L.family_vocab(fam, lvl, Str, cache_m)
                    c = L.translation_certificate(Ste, Z, wm, voc)
                    row = dict(dataset=name, group=group, split=sp, model=mname, family=fam, level=lvl,
                               rbar=c["rbar"], cert=float(c["cert"].mean()), cert_static=float(c["cert_static"].mean()),
                               changed=float(c["changed"].mean()), viol=int((c["cert"] & c["changed"]).sum()),
                               viol_static=int((c["cert_static"] & c["changed"]).sum()), n=len(Ste))
                    if mname == "KM":                       # how the certificate relates to GW-HAC itself
                        *_, lab_gw = L.word_model(Str, Ste, voc, K, sp, wm)
                        m = L.match_labels(lab_gw, c["g"], K)
                        row.update(ari_gw_vs_numeric=float(ARI(c["g"], lab_gw)), agree_gw=float(np.mean(m == c["g"])),
                                   agree_gw_on_cert=float(np.mean(m[c["cert"]] == c["g"][c["cert"]])) if c["cert"].any() else np.nan,
                                   agree_gw_off_cert=float(np.mean(m[~c["cert"]] == c["g"][~c["cert"]])) if (~c["cert"]).any() else np.nan)
                    cert.append(row)
        # ---- (d) certified granularity ---------------------------------------------------------------------------
        for lvl in (3, 4, 5):
            voc = L.family_vocab("HA-grid", lvl, S_tr, cache)
            f, hits, q_tr, q_te, lab = L.word_model(S_tr, S_te, voc, K, sp, w)
            mstar, per_m = L.certified_granularity(q_te, voc, f.P, w, lvl,
                                                   lambda l: L.family_vocab("HA-grid", l, S_tr, cache))
            for m in range(lvl):
                gran.append(dict(dataset=name, group=group, split=sp, level=lvl, kind="mstar_share", m=m,
                                 value=float(np.mean(mstar == m))))
            for pm in per_m:
                gran.append(dict(dataset=name, group=group, split=sp, level=lvl, kind="per_m", **pm))
    pd.DataFrame(feat).to_csv(raw / f"v2feat_{name}.csv", index=False)
    pd.DataFrame(dec).to_csv(raw / f"v2dec_{name}.csv", index=False)
    pd.DataFrame(cert).to_csv(raw / f"v2cert_{name}.csv", index=False)
    pd.DataFrame(gran).to_csv(raw / f"v2gran_{name}.csv", index=False)
    return f"[done] {name} ({group}) in {time.time() - t0:.0f}s"


def main() -> None:
    ap = L.common_args(__doc__)
    ap.add_argument("--no-ewa", action="store_true", help="skip the EWA-HAC translation certificate (slow)")
    a = ap.parse_args()
    L.run_pool(run_dataset, [(n, g, Path(a.out), not a.no_ewa) for n, g in L.jobs_for(a.groups, a.only)], a.workers)


if __name__ == "__main__":
    main()
