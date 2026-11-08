"""Analysis of the v2 development runs (60-64): JSON + CSV + LaTeX tables in <out>/tables.

All summaries use one value per dataset (mean over splits) and medians over datasets; paired differences carry a
bootstrap 95 % CI over datasets and a two-sided Wilcoxon p-value.  These analyses are exploratory.

    python v2/69_dev_analyze.py [--groups dev] [--out ...]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import v2lib as L

B = L.CFG["bootstrap"]["B"]
SEED = L.CFG["bootstrap"]["seed"]


def boot_ci(d, B=B, seed=SEED):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(d[rng.integers(0, len(d), (B, len(d)))], 1)
    return float(np.quantile(meds, .025)), float(np.quantile(meds, .975))


def paired(a: pd.Series, b: pd.Series):
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (j.a - j.b).to_numpy()
    if len(d) == 0:
        return dict(n=0)
    p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) and len(d) > 1 else 1.0
    lo, hi = boot_ci(d)
    return dict(n=int(len(d)), median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi, p_two_sided=p,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def read(raw: Path, prefix: str, groups):
    fs = sorted(glob.glob(str(raw / f"{prefix}_*.csv")))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df[df.group.isin(groups)] if "group" in df else df


def tex_table(df: pd.DataFrame, path: Path, floatfmt="%.3f"):
    path.write_text(df.to_latex(index=False, float_format=floatfmt, na_rep="--"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.RESULTS))
    ap.add_argument("--groups", nargs="*", default=["dev"])
    a = ap.parse_args()
    out = Path(a.out)
    raw, tab = out / "raw", out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    sfx = "_".join(a.groups)
    summary = {"groups": a.groups}

    # ---------------- E1 / E2: factorial and baselines -----------------------------------------------------------
    fac = read(raw, "v2fac", a.groups)
    if len(fac):
        ds = fac.groupby(["dataset", "method", "level"]).ari.mean().unstack(["method", "level"])
        prim = ds[(L.tag(L.PRIMARY), L.MAIN_LEVEL)]
        rows = []
        for (meth, lvl) in ds.columns:
            r = dict(method=meth, level=lvl, median_ari=float(ds[(meth, lvl)].median()), n=int(ds[(meth, lvl)].notna().sum()))
            if lvl > 0:
                ref = ds[(L.tag(L.PRIMARY), lvl)]
                r.update({f"vsGWHAC_same_level_{k}": v for k, v in paired(ds[(meth, lvl)], ref).items()})
            r.update({f"GWHAC3_minus_{k}": v for k, v in paired(prim, ds[(meth, lvl)]).items()})
            rows.append(r)
        T = pd.DataFrame(rows).sort_values(["level", "median_ari"], ascending=[True, False])
        T.to_csv(tab / f"v2_factorial_{sfx}.csv", index=False)
        main_rows = T[(T.level.isin([0, L.MAIN_LEVEL]))][["method", "level", "median_ari", "GWHAC3_minus_median_diff",
                                                         "GWHAC3_minus_ci_lo", "GWHAC3_minus_ci_hi",
                                                         "GWHAC3_minus_p_two_sided", "GWHAC3_minus_wins", "n"]]
        tex_table(main_rows, tab / f"v2_baselines_{sfx}.tex")
        # main effects of each factor (HA families only, levels 2-5): median over datasets of paired differences
        hf = fac[fac.family.isin(["dyadic", "HA-theta", "HA-grid"])]
        dsm = hf.groupby(["dataset", "family", "rep", "rule", "level"]).ari.mean()
        eff = {}
        for factor, ref, alts in (("rule", "interval", ["nearest"]), ("rep", "native", ["mid", "mean"]),
                                  ("family", "HA-grid", ["dyadic", "HA-theta"])):
            u = dsm.unstack(factor)
            for alt in alts:
                for lvl in L.LEVELS:
                    sub = u.xs(lvl, level="level")
                    eff[f"{factor}:{alt}-{ref}:l{lvl}"] = paired(sub[alt], sub[ref])
        summary["factor_effects"] = eff
        summary["cap_hits_total"] = int(fac.cap_hits.fillna(0).sum())
        summary["fits_total"] = int(fac.cap_hits.notna().sum())
        summary["GWHAC3_vs"] = {m: paired(prim, ds[(m, 0)]) for (m, l) in ds.columns if l == 0}
        summary["GWHAC3_median_ari"] = float(prim.median())

    # ---------------- E3 a: Medium, hedge asymmetry, price ---------------------------------------------------------
    ft = read(raw, "v2feat", a.groups)
    price = read(raw, "v2price", a.groups)
    if len(ft):
        lo = np.minimum(ft["median"], .5) - 1e-3
        hi = np.maximum(ft["median"], .5) + 1e-3
        th = dict(share_between_median_and_half=float(((ft.theta_hat >= lo) & (ft.theta_hat <= hi)).mean()))
        band = L.CFG["alpha_band"]
        per = []
        for lvl, g in ft.groupby("level"):
            inb = ((g.alpha_grid_l >= band[0]) & (g.alpha_grid_l <= band[1]))
            sel = (g.alpha_given_theta_hat - .5).abs() >= 0.05
            agree = (np.sign(g.alpha_given_theta_hat - .5) == -np.sign(g.Q_hat))[sel]
            dsl = g.groupby("dataset").mean(numeric_only=True)
            r = dict(level=int(lvl), alpha_in_band=float(inb.mean()),
                     sign_agree_alpha_Q=float(agree.mean()) if len(agree) else np.nan, n_sign=int(sel.sum()))
            for fam in ("HA-theta", "HA-grid", "HA-grid-l"):
                red = 1 - dsl[f"mse_te_{fam}"] / dsl["mse_te_dyadic"]
                r[f"heldout_mse_reduction_{fam}_vs_dyadic"] = float(red.median())
            if len(price):                               # corrected, exact reference (61b); else the 61 column
                gp = price[price.level == lvl]
                for fam in ("dyadic", "HA-theta", "HA-grid", "HA-grid-l"):
                    pr = gp[f"price_{fam}"].groupby(gp.dataset).median()
                    r[f"price_median_{fam}"] = float(pr.median())
                r["price_min_any"] = float(gp[[f"price_{f}" for f in ("dyadic", "HA-theta", "HA-grid", "HA-grid-l")]].min().min())
            else:
                for fam in ("dyadic", "HA-theta", "HA-grid", "HA-grid-l"):
                    pr = (g[f"mse_tr_{fam}"] / g.mse_lloyd_tr.clip(lower=1e-300)).groupby(g.dataset).median()
                    r[f"price_median_{fam}"] = float(pr.median())
            per.append(r)
        th["per_level"] = per
        bandshare = ft.assign(inb=(ft.alpha_grid_l >= band[0]) & (ft.alpha_grid_l <= band[1])).groupby(
            ["dataset", "level"]).inb.mean().unstack()
        th["P2_dev_sign_test_l5_vs_l2"] = dict(more=int((bandshare[5] > bandshare[2]).sum()),
                                               less=int((bandshare[5] < bandshare[2]).sum()))
        summary["theory_features"] = th
        tex_table(pd.DataFrame(per), tab / f"v2_theory_features_{sfx}.tex")

    # ---------------- E3 b: decomposition ---------------------------------------------------------------------------
    dec = read(raw, "v2dec", a.groups)
    if len(dec):
        dec = dec.assign(rel_gap=(dec.Jl - dec.JN) / dec.JN, nd_rel=dec.ND / dec.JN,
                         t2_share=dec.T2.abs() / (dec.T1 + dec.T3).clip(lower=1e-300))
        d = dec.groupby(["dataset", "family", "level"]).mean(numeric_only=True).reset_index()
        rows = []
        for (fam, lvl), g in d.groupby(["family", "level"]):
            rows.append(dict(family=fam, level=int(lvl), median_rel_gap=float(g.rel_gap.median()),
                             median_abs_rel_gap=float(g.rel_gap.abs().median()), median_ND_over_JN=float(g.nd_rel.median()),
                             median_T2_share=float(g.t2_share.median()), median_gap_fit=float(g.gap_fit.median())))
        tex_table(pd.DataFrame(rows), tab / f"v2_decomposition_{sfx}.tex", "%.4f")
        slopes = {}
        for fam, g in d.groupby("family"):
            s = []
            for dsn, h in g.groupby("dataset"):
                h = h.sort_values("level")
                s.append(np.polyfit(h.level, np.log2(h.rel_gap.abs() + 1e-6), 1)[0])
            slopes[fam] = dict(median_slope_log2_absgap_per_level=float(np.median(s)), mean=float(np.mean(s)),
                               n=len(s))
        summary["decomposition"] = dict(identity_max_err=float(dec.identity_err.max()), per=rows, slopes=slopes)

    # ---------------- E3 c, d: certificates and granularity --------------------------------------------------------
    cert = read(raw, "v2cert", a.groups)
    if len(cert):
        c = cert.groupby(["dataset", "model", "family", "level"]).mean(numeric_only=True).reset_index()
        cols = ["cert", "cert_static", "changed", "agree_gw", "agree_gw_on_cert", "agree_gw_off_cert", "ari_gw_vs_numeric"]
        ct = c.groupby(["model", "family", "level"])[[k for k in cols if k in c]].median().reset_index()
        ct["violations_total"] = cert.groupby(["model", "family", "level"]).viol.sum().values
        ct["violations_static_total"] = cert.groupby(["model", "family", "level"]).viol_static.sum().values
        tex_table(ct, tab / f"v2_certificates_{sfx}.tex")
        summary["certificates"] = ct.to_dict("records")
    gran = read(raw, "v2gran", a.groups)
    if len(gran):
        gs = gran[gran.kind == "mstar_share"].groupby(["dataset", "level", "m"]).value.mean().groupby(["level", "m"]).median()
        pm = gran[gran.kind == "per_m"].groupby(["level", "m"])[["cert", "keep"]].median()
        pm["viol_total"] = gran[gran.kind == "per_m"].groupby(["level", "m"]).viol.sum()
        g2 = pm.join(gs.rename("median_share_mstar_eq_m"), how="outer").reset_index()
        tex_table(g2, tab / f"v2_granularity_{sfx}.tex")
        summary["granularity"] = g2.to_dict("records")

    # ---------------- E4: interpretability proxies -----------------------------------------------------------------
    it = read(raw, "v2int", a.groups)
    if len(it):
        ii = it.groupby(["dataset", "kind", "method", "level"]).value.mean().groupby(["kind", "method", "level"]).median().reset_index()
        tex_table(ii, tab / f"v2_interp_{sfx}.tex")
        summary["interp"] = ii.to_dict("records")

    # ---------------- E5, E6 --------------------------------------------------------------------------------------
    sk = read(raw, "v2skw", a.groups)
    if len(sk):
        s = sk.groupby(["dataset", "group", "prep", "method"]).ari.mean().unstack(["prep"]).reset_index()
        s.to_csv(tab / f"v2_skew_per_dataset_{sfx}.csv", index=False)
        diag = sk.groupby(["dataset"])[["share_top90", "mean_abs_skew", "share_atmin30"]].mean()
        res = {}
        for meth, g in s.groupby("method"):
            res[meth] = {f"{p}_minus_minmax": paired(g.set_index("dataset")[p], g.set_index("dataset")["minmax"])
                         for p in ("log1p", "rank")}
        gw = s[s.method == "GW-HAC"].set_index("dataset")
        fl = s[s.method == "FL-U8"].set_index("dataset")
        res["GW_minus_FLU8_minmax_vs_share_top90_spearman"] = float(pd.concat(
            [gw.minmax - fl.minmax, diag.share_top90], axis=1).corr(method="spearman").iloc[0, 1])
        summary["skew"] = res
        diag.to_csv(tab / f"v2_skew_diagnostics_{sfx}.csv")
    no = read(raw, "v2noi", a.groups)
    if len(no):
        n = no.groupby(["dataset", "method", "sigma"]).flip.mean().groupby(["method", "sigma"]).median().unstack()
        n.to_csv(tab / f"v2_noise_{sfx}.csv")
        tex_table(n.reset_index(), tab / f"v2_noise_{sfx}.tex")
        summary["noise_median_flip"] = n.round(5).to_dict()

    (tab / f"v2_dev_summary_{sfx}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote tables to {tab}")


if __name__ == "__main__":
    main()
