"""Pre-registered analysis of the confirmatory study (PREREG_E7_FILLED.md, Section 5).

Family A (Holm, one-sided family alpha 0.025): H1 NI vs FL-U8, H2 NI vs k-means, H3 TOST vs GW-nw (margin delta).
Family B (Holm, 0.05): P1 mean per-dataset slope of log2(|gap|/J_N + 1e-6) on the level < -1; P2 sign test on the
share of alpha_hat in [0.45, 0.55] at level 5 versus 2.
In "estimation" mode (fewer datasets than required) Family A is reported without decisions.
Secondary (descriptive): log1p preprocessing, GW-HAC - FL-U8 and GW-HAC(log1p) - GW-HAC(minmax).
Sensitivity (secondary): Family A again without datasets where every primary method has ARI < 0.02 (S1) and
without OpenML ids 700-999, the binarized-regression collection (S2).

Output: <out>/tables/confirm.json, confirm_table.tex, confirm_per_dataset.csv

    python v2/74_confirm_analyze.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

import v2lib as L

BOOT = L.CFG["bootstrap"]


def boot_ci(d):
    rng = np.random.default_rng(BOOT["seed"])
    m = np.median(d[rng.integers(0, len(d), (BOOT["B"], len(d)))], 1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def wil(x, alt):
    x = np.asarray(x, float)
    if np.all(x == 0):
        return 1.0
    return float(wilcoxon(x, zero_method="zsplit", alternative=alt).pvalue)


def family_a(per: pd.DataFrame, delta: float, alpha_A: float, decide: bool) -> list:
    """H1-H3 with Holm (one-sided, family-wise alpha_A); H3 p = max of the two TOST p-values."""
    out = []
    for hid, comp, kind in (("H1", "FL-U8", "NI"), ("H2", "KM", "NI"), ("H3", "GW-nw", "TOST")):
        d = (per["GW-HAC"] - per[comp]).dropna().to_numpy()
        lo, hi = boot_ci(d)
        r = dict(id=hid, comparator=comp, kind=kind, n=len(d), median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi,
                 wins=int((d > 0).sum()), losses=int((d < 0).sum()))
        if kind == "NI":
            r["p"] = wil(d + delta, "greater")
        else:
            r["p_lower"], r["p_upper"] = wil(d + delta, "greater"), wil(d - delta, "less")
            r["p"] = max(r["p_lower"], r["p_upper"])
        out.append(r)
    if decide:
        ok = True
        for i, r in enumerate(sorted(out, key=lambda r: r["p"])):
            r["holm_threshold"] = alpha_A / (len(out) - i)
            r["tested"] = True
            r["rejected"] = bool(ok and r["p"] < r["holm_threshold"])
            ok = r["rejected"]
    else:
        for r in out:
            r["tested"], r["rejected"] = False, None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.RESULTS))
    a = ap.parse_args()
    out = Path(a.out)
    raw, tab = out / "raw", out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    sel = json.loads((L.VAL2_DIR / "SELECTION.json").read_text())
    lock = json.loads((L.VAL2_DIR / "SELECTION_LOCK.json").read_text())
    delta, mode = float(sel["power"]["delta"]), sel["mode"]
    alpha_A, alpha_B = L.CFG["familyA_alpha"], L.CFG["familyB_alpha"]
    conf = pd.concat([pd.read_csv(f) for f in glob.glob(str(raw / "v2conf_*.csv"))], ignore_index=True)
    per = conf.groupby(["dataset", "method"]).ari.mean().unstack()
    per.to_csv(tab / "confirm_per_dataset.csv")
    n = int(per["GW-HAC"].notna().sum())
    if n < sel["power"]["n_required"]:
        mode = "estimation"
    res = dict(osf_id=lock.get("osf_id"), delta=delta, mode=mode, n_datasets=n, n_required=sel["power"]["n_required"],
               familyA=[], familyB=[], cap_hits=int(conf.get("cap_hits", pd.Series(dtype=float)).fillna(0).sum()))
    res["familyA"] = family_a(per, delta, alpha_A, mode == "test")
    # registered sensitivity analyses (secondary; they do not change the Family A decisions)
    prim = ["GW-HAC", "FL-U8", "KM", "GW-nw"]
    man = json.loads((L.VAL2_DIR / "MANIFEST.json").read_text())["datasets"] if (L.VAL2_DIR / "MANIFEST.json").exists() else {}
    ids = {k: int(v.get("id", -1)) for k, v in man.items()}
    s1 = per[per[prim].max(axis=1) >= L.CFG.get("sensitivity", {}).get("S1_min_ari", 0.02)]
    s2 = per[[not (700 <= ids.get(k, -1) <= 999) for k in per.index]]
    res["sensitivity"] = {
        "S1_drop_uninformative": dict(n_dropped=int(len(per) - len(s1)), familyA=family_a(s1, delta, alpha_A, True)),
        "S2_drop_ids_700_999": dict(n_dropped=int(len(per) - len(s2)), familyA=family_a(s2, delta, alpha_A, True))}
    sec = {}
    for name, a_, b_ in (("GW-HAC_minus_FL-U8_log1p", "GW-HAC|log1p", "FL-U8|log1p"),
                         ("GW-HAC_log1p_minus_minmax", "GW-HAC|log1p", "GW-HAC"),
                         ("GW-HAC_minus_KM_log1p", "GW-HAC|log1p", "KM|log1p")):
        if a_ in per and b_ in per:
            d = (per[a_] - per[b_]).dropna().to_numpy()
            lo, hi = boot_ci(d)
            sec[name] = dict(n=len(d), median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi,
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()))
    res["secondary_log1p"] = sec
    dec = pd.concat([pd.read_csv(f) for f in glob.glob(str(raw / "v2confdec_*.csv"))], ignore_index=True)
    dec["absgap"] = ((dec.Jl - dec.JN) / dec.JN).abs()
    dd = dec.groupby(["dataset", "level"])[["absgap", "T1", "T2", "T3", "ND", "JN"]].mean().reset_index()
    slopes = np.array([np.polyfit(g.level, np.log2(g.absgap + 1e-6), 1)[0] for _, g in dd.groupby("dataset")])
    t = ttest_1samp(slopes, -1.0, alternative="less")
    p1 = dict(id="P1", n=len(slopes), mean_slope=float(slopes.mean()), median_slope=float(np.median(slopes)),
              p=float(t.pvalue), median_T2_share=float((dd.T2.abs() / (dd.T1 + dd.T3)).median()))
    al = pd.concat([pd.read_csv(f) for f in glob.glob(str(raw / "v2confalpha_*.csv"))], ignore_index=True)
    sh = al.groupby(["dataset", "level"]).share_in_band.mean().unstack()
    more, less = int((sh[5] > sh[2]).sum()), int((sh[5] < sh[2]).sum())
    p2 = dict(id="P2", n=more + less, more=more, less=less, median_share_l2=float(sh[2].median()),
              median_share_l5=float(sh[5].median()),
              p=float(binomtest(more, more + less, 0.5, alternative="greater").pvalue) if more + less else 1.0)
    ps = sorted([p1, p2], key=lambda r: r["p"])                              # Holm
    for i, r in enumerate(ps):
        r["holm_threshold"] = alpha_B / (len(ps) - i)
        r["rejected"] = bool(r["p"] < r["holm_threshold"] and (i == 0 or ps[i - 1]["rejected"]))
    res["familyB"] = [p1, p2]
    (tab / "confirm.json").write_text(json.dumps(res, indent=1))
    lines = [r"\begin{tabular}{lllrrrrl}", r"\toprule",
             r"Claim & Comparator & Test & $n$ & median diff. & 95\% CI & $p$ & Decision\\", r"\midrule"]
    for r in res["familyA"]:
        dec_ = ("estimation only" if r["rejected"] is None else "supported" if r["rejected"] else
                "not supported")
        lines.append(f"{r['id']} & {r['comparator']} & {r['kind']} ($\\delta={delta:.3f}$) & {r['n']} & "
                     f"{r['median_diff']:+.3f} & [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {r['p']:.3f} & {dec_}\\\\")
    lines.append(f"P1 & -- & slope $<-1$ & {p1['n']} & {p1['mean_slope']:.2f} & -- & {p1['p']:.3f} & "
                 f"{'supported' if p1['rejected'] else 'not supported'}\\\\")
    lines.append(f"P2 & -- & sign test & {p2['n']} & {p2['more']}/{p2['less']} & -- & {p2['p']:.3f} & "
                 f"{'supported' if p2['rejected'] else 'not supported'}\\\\")
    for sname, sv in res["sensitivity"].items():
        for r in sv["familyA"]:
            lines.append(f"{sname.split('_')[0]}: {r['id']} & {r['comparator']} & {r['kind']} & {r['n']} & {r['median_diff']:+.3f} & "
                         f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {r['p']:.3f} & {'supported' if r['rejected'] else 'not supported'}\\\\")
    for k, r in sec.items():
        lines.append(f"{k.replace('_', ' ')} & -- & descriptive & {r['n']} & {r['median_diff']:+.3f} & "
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & -- & --\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "confirm_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
