"""Check the regenerated outputs against the numbers printed in the IEEE TETCI submission
(TETCI-2026-3160, "Distortion-Bounded Linguistic Representation for Interpretable Multigranular Clustering").

    python tools/check_tetci.py [out]          (default: out/; run `sh reproduce.sh tables numerics` first)

tools/check_outputs.py already checks Table I, Fig. 1, Table IV and byte identity with the archive. This script adds
every other number of Sections V-A to V-D: the Gaussian decomposition example, Tables II, III and V, the certificates,
the descriptions, noise, Spambase, the vocabulary price on real features, Fig. 2 and the running times.
Rounding follows the paper: a value v printed with k decimals passes if round(v, k) equals it.
Exit status 1 if any check fails.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out"
T = OUT / "tables"
RAW = ROOT / "results" / "archive" / "raw"
fails, errata, n = [], [], 0


def check(ok, msg):
    global n
    n += 1
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


def check_print(exact, near, msg, correct):
    """A printed number that is off by one unit in its last digit (double rounding in the submission) passes,
    marked ok*, and is listed as an erratum with the correctly rounded value."""
    global n
    n += 1
    tag = "ok    " if exact else ("ok*   " if near else "FAIL  ")
    print(tag + msg)
    if not exact and near:
        errata.append(f"{msg}  ->  correct: {correct}")
    if not (exact or near):
        fails.append(msg)


def eq(v, printed, k):
    return round(float(v), k) == printed or (printed == 0 and abs(round(float(v), k)) == 0)


def pct(v):
    return round(100 * float(v))


def sci(p):                                       # 7e-4 style: one significant digit
    m, e = f"{p:.0e}".split("e")
    return f"{m}e{int(e)}"


dev = json.loads((T / "v2_dev_summary_dev.json").read_text())
val11 = json.loads((T / "v2_dev_summary_val11.json").read_text())
num = json.loads((T / "numerics_sec6.json").read_text())
conf = json.loads((T / "confirm.json").read_text())

# ---- V-A: four Gaussian clusters, Proposition 6 ------------------------------------------------------------------
dec = num["decomposition"]
t2 = [r["T2_ratio"] for r in dec]
gr = [r["gap_over_ND"] for r in dec]
check(pct(max(t2)) == 13 and pct(np.median(t2)) == 2, f"Gauss example: |T2|/(T1+T3) max {max(t2):.3f} (13%), median {np.median(t2):.3f} (2%)")
check(round(min(gr), 1) == 1.6 and round(max(gr), 1) == 2.4, f"Gauss example: relative gap / (N D/J_N) in [{min(gr):.2f}, {max(gr):.2f}] (1.6-2.4)")

# ---- V-C a: Table II ---------------------------------------------------------------------------------------------
TAB2 = {  # comparator: (dev median, lo, hi, p, wins, losses), (val11 ...)
    "FL-U8": ((0.002, -0.019, 0.046, "0.60", 13, 12), (-0.020, -0.070, 0.012, "0.08", 3, 8)),
    "FL-U15": ((0.041, -0.006, 0.092, "0.045", 16, 9), (0.006, -0.081, 0.047, "0.83", 7, 4)),
    "KM": ((0.004, -0.010, 0.009, "1.00", 14, 11), (0.000, -0.006, 0.014, "0.83", 6, 5)),
    "GW-nw": ((-0.005, -0.012, 0.010, "0.56", 12, 13), (-0.002, -0.003, 0.012, "0.83", 4, 7)),
    "IMM": ((0.016, 0.001, 0.036, "0.039", 18, 7), (0.016, 0.004, 0.051, "0.024", 10, 1)),
    "ExKMC": ((0.010, 0.006, 0.014, "0.002", 19, 6), (0.008, 0.002, 0.042, "0.032", 9, 2)),
}
for comp, rows in TAB2.items():
    for name, s, (md, lo, hi, p, w, l) in (("dev", dev, rows[0]), ("val11", val11, rows[1])):
        r = s["GWHAC3_vs"][comp]
        dp = len(p.split(".")[1])
        ok = (eq(r["median_diff"], md, 3) and eq(r["ci_lo"], lo, 3) and eq(r["ci_hi"], hi, 3)
              and f"{r['p_two_sided']:.{dp}f}" == p and r["wins"] == w and r["losses"] == l)
        check(ok, f"Table II {comp} {name}: {r['median_diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                  f"p {r['p_two_sided']:.3f} {r['wins']}/{r['losses']}")
check(eq(dev["GWHAC3_median_ari"], 0.247, 3), f"median GW-HAC ARI {dev['GWHAC3_median_ari']:.3f} (0.247)")
check((dev["cap_hits_total"], dev["fits_total"]) == (119, 11132), f"iteration cap {dev['cap_hits_total']} of {dev['fits_total']} fits")

# ---- Table III and the in-text theory checks ---------------------------------------------------------------------
th = {r["level"]: r for r in dev["theory_features"]["per_level"]}
rows3 = {
    "alpha_in_band": [16, 44, 59, 74],
    "sign_agree_alpha_Q": [72, 94, 98, 98],
    "heldout_mse_reduction_HA-theta_vs_dyadic": [19, 16, 17, 18],
    "heldout_mse_reduction_HA-grid-l_vs_dyadic": [32, 27, 28, 29],
}
for key, want in rows3.items():
    got = [pct(th[l][key]) for l in (2, 3, 4, 5)]
    check_print(got == want, all(abs(a - b) <= 1 for a, b in zip(got, want)), f"Table III {key}: paper {want}", f"{got} %")
for key, want in (("price_median_dyadic", [1.69, 1.78, 2.21, 3.72]), ("price_median_HA-grid-l", [1.25, 1.52, 1.87, 2.64])):
    got = [round(th[l][key], 2) for l in (2, 3, 4, 5)]
    check_print(got == want, all(abs(a - b) <= 0.0100001 for a, b in zip(got, want)), f"Table III {key}: paper {want}", got)
dper = {(r["family"], r["level"]): r for r in dev["decomposition"]["per"]}
got = [round(dper[("HA-grid", l)]["median_abs_rel_gap"], 3) for l in (2, 3, 4, 5)]
check(got == [0.465, 0.103, 0.028, 0.008], f"Table III median |J_l-J_N|/J_N (HA-grid): {got}")
got = [round(dper[("HA-grid", l)]["median_T2_share"], 2) for l in (2, 3, 4, 5)]
check(got == [0.14, 0.24, 0.47, 0.82], f"Table III median |T2|/(T1+T3) (HA-grid): {got}")
cert = {(r["model"], r["family"], r["level"]): r for r in dev["certificates"]}
got = [pct(cert[("EWA-HAC", "HA-grid", l)]["cert"]) for l in (2, 3, 4, 5)]
check(got == [13, 57, 82, 89], f"Table III translation certificate (EWA-HAC): {got}")
got = [round(100 * cert[("EWA-HAC", "HA-grid", l)]["cert_static"], 1) for l in (2, 3, 4, 5)]
check(got == [0.0, 0.0, 0.7, 39.3], f"Table III data-independent condition: {got} (printed 0%, 0%, 0.7%, 39%)")
check(all(r["violations_total"] == 0 for r in dev["certificates"]), "no certified assignment changed (Theorem 4)")
th_red = [th[l]["heldout_mse_reduction_HA-theta_vs_dyadic"] / th[l]["heldout_mse_reduction_HA-grid-l_vs_dyadic"] for l in (2, 3, 4, 5)]
check(0.55 <= np.mean(th_red) <= 0.65, f"theta alone captures about 60% of the grid reduction: {np.mean(th_red):.2f}")
check(dev["theory_features"]["share_between_median_and_half"] == 1.0, "every theta_j between the empirical median and 1/2")
p2 = dev["theory_features"]["P2_dev_sign_test_l5_vs_l2"]
check(p2 == {"more": 25, "less": 0}, f"calibrated alpha approached 1/2 on all 25 datasets: {p2}")
k3 = cert[("KM", "HA-grid", 3)]
check(pct(k3["agree_gw_on_cert"]) == 99 and pct(k3["agree_gw_off_cert"]) == 79,
      f"translated k-means: agreement {k3['agree_gw_on_cert']:.3f} certified, {k3['agree_gw_off_cert']:.3f} uncertified")
gran = {(r["level"], r["m"]): r for r in dev["granularity"]}
check(pct(gran[(3, 0)]["median_share_mstar_eq_m"]) == 91 and pct(gran[(3, 1)]["keep"]) == 93 and pct(gran[(3, 2)]["keep"]) == 84,
      "Corollary 2 at level 3: 91% m*=0, 93% and 84% kept one and two levels down")
v = gran[(5, 1)]["cert"]
check_print(pct(v) == 75, abs(pct(v) - 75) <= 1, "level 5: certified one level down, paper 75%", f"{100 * v:.2f}% -> {pct(v)}%")

# decomposition on real data (per-split raw results)
d = pd.concat(pd.read_csv(f) for f in glob.glob(str(RAW / "v2dec_*.csv")))
check(d.identity_err.max() < 3e-12, f"Proposition 6 residual {d.identity_err.max():.1e} < 3e-12")
g = d.groupby(["dataset", "family", "level"]).mean(numeric_only=True).reset_index()
s = [np.polyfit(h.level, np.log2((h.Jl - h.JN).abs()), 1)[0] for _, h in g[g.family == "HA-theta"].groupby("dataset")]
check(round(np.mean(s), 2) == -1.82, f"absolute gap: mean log2 slope {np.mean(s):.3f} (-1.82; HA-theta, per dataset)")
neg = (d[d.family == "HA-grid"].T2 < 0).mean()
check(pct(neg) == 85, f"T2 negative in {100 * neg:.1f}% of cases (85%; HA-grid, per split)")

# vocabulary price on near-continuous features
p = pd.concat(pd.read_csv(f) for f in glob.glob(str(RAW / "v2price_*.csv")))
p = p[(p.group == "dev") & (p.n_unique >= 64 * 2 ** p.level)]
rec = ((p.price_dyadic - p["price_HA-theta"]) / (p.price_dyadic - 1)).groupby(p.level).median()
check((pct(rec.min()), pct(rec.max())) == (44, 59), f"near-continuous features: HA-theta recovers {pct(rec.min())}-{pct(rec.max())}% (44-59%)")

# ---- descriptions, noise, preprocessing --------------------------------------------------------------------------
it = {(r["kind"], r["method"], r["level"]): r["value"] for r in dev["interp"]}
check(pct(it[("fidelity", "KM+HA-words", 3)]) == 97 and pct(it[("fidelity", "FL-U8", 3)]) == 86,
      "post hoc naming reproduces 97% (k-means + HA words) and 86% (FL-U8 winning labels)")
got = tuple(round(it[k], 2) for k in (("fidelity", "GW-HAC", 2), ("fidelity", "KM+HA-words", 2),
                                      ("fidelity", "GW-HAC", 1), ("fidelity", "KM+HA-words", 1)))
check(got == (0.93, 0.92, 0.83, 0.83), f"levels 2 and 1: {got} (0.93 vs 0.92, 0.83 vs 0.83)")
check(round(it[("desc_words_per_cluster", "GW-HAC", 3)]) == 12, "prototype differs from the modal word in 12 features")
check(round(it[("tree_depth", "IMM", 0)], 1) == 2.2 and round(it[("tree_depth", "ExKMC", 0)], 1) == 3.2, "tree depth 2.2 (IMM), 3.2 (ExKMC)")
nz = dev["noise_median_flip"]["0.01"]
check(round(100 * nz["GW-HAC"], 1) == 1.4 and round(100 * nz["KM"], 1) == 0.5, f"noise 0.01: {nz['GW-HAC']:.4f} GW-HAC, {nz['KM']:.4f} k-means")
sk = pd.read_csv(T / "v2_skew_per_dataset_val11.csv")
sk = sk[sk.dataset == "spambase"].set_index("method")
ok = (eq(sk.loc["GW-HAC", "minmax"], 0.07, 2) and eq(sk.loc["KM", "minmax"], 0.06, 2)
      and eq(sk.loc["GW-HAC", "log1p"], 0.48, 2) and eq(sk.loc["GW-HAC", "rank"], 0.44, 2))
check(ok, "Spambase: GW-HAC 0.07, k-means 0.06 (min-max); GW-HAC 0.48 with log(1+x), 0.44 with rank scaling")

# ---- running time ------------------------------------------------------------------------------------------------
tn = (OUT / "timing_numbers.txt").read_text()
tab = {(m.group(1), m.group(2).strip()): (float(m.group(3)), float(m.group(4)))
       for m in re.finditer(r"^\s*(dev|val2)\s+(.+?)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+[\d.]+$", tn, re.M)}
ok = (round(tab[("dev", "KM")][0], 1) == 5.5 and round(tab[("dev", "FL-U8")][0]) == 10
      and round(tab[("dev", "fit_gwhac")][0], 1) == 6.2)
check(ok, "dev medians: 5.5 ms k-means, 10 ms FL-U8, 6.2 ms fitting")
v = tab[("dev", "GW-HAC total")][0]
check_print(int(v + 0.5) == 65, abs(v - 65) <= 1, "dev median total GW-HAC (HA-grid): paper 65 ms (Sec. V-C and V-D)", f"{v} ms -> 66 ms")
check(round(tab[("dev", "GW-HAC (HA-theta)")][1], 1) == 1.3 and round(tab[("val2", "GW-HAC (HA-theta)")][1], 1) == 1.4,
      "HA-theta total runtime 1.3x (dev) and 1.4x (confirmatory) k-means")
sh = [float(x) for x in re.findall(r"median calibration share ([\d.]+)%", tn)]
check([round(x) for x in sh] == [91, 94], f"calibration share {sh} (91%, 94%)")
check("har N=10299 F=561 total 18.6 s (calibration 15.2 s, k-means 0.69 s)" in tn, "HAR: 18.6 s, 15.2 s calibration, 0.7 s k-means")
check("N=50000 F=21 total 2.7 s" in tn, "largest confirmatory dataset (N = 50 000, d = 21): 2.7 s")

# ---- V-C b: confirmatory study, Table V, Fig. 2 ------------------------------------------------------------------
A = {r["id"]: r for r in conf["familyA"]}
B = {r["id"]: r for r in conf["familyB"]}
S1 = {r["id"]: r for r in conf["sensitivity"]["S1_drop_uninformative"]["familyA"]}
S2 = {r["id"]: r for r in conf["sensitivity"]["S2_drop_ids_700_999"]["familyA"]}
check(conf["n_datasets"] == 126 and conf["n_required"] == 110 and conf["cap_hits"] == 0, "126 datasets >= n = 110, no iteration cap")
for h, lo, hi, ps in (("H1", -0.003, 0.000, "7e-4"), ("H2", -0.003, 0.000, "3e-9"), ("H3", -0.000, 0.000, "1e-11")):
    r = A[h]
    check(eq(r["median_diff"], 0.0, 3) and eq(r["ci_lo"], lo, 3) and eq(r["ci_hi"], hi, 3) and sci(r["p"]) == ps and r["rejected"],
          f"Table V {h}: {r['median_diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p {r['p']:.1e} supported")
check(eq(B["P1"]["mean_slope"], -1.45, 2) and sci(B["P1"]["p"]) == "7e-10" and B["P1"]["rejected"], f"Table V P1: mean {B['P1']['mean_slope']:.3f}")
check((B["P2"]["n"], B["P2"]["more"], B["P2"]["less"]) == (117, 112, 5) and sci(B["P2"]["p"]) == "1e-27", "Table V P2: 112 vs 5 of 117")
r = S1["H1"]
check(r["n"] == 88 and eq(r["median_diff"], 0.0, 3) and eq(r["ci_lo"], -0.018, 3) and eq(r["ci_hi"], 0.008, 3)
      and round(r["p"], 3) == 0.048 and not r["rejected"] and (r["wins"], r["losses"]) == (42, 39),
      f"Table V S1 H1: n 88, [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], p {r['p']:.3f}, not supported, 42/39")
check(eq(S1["H2"]["median_diff"], -0.002, 3) and sci(S1["H2"]["p"]) == "1e-4" and sci(S1["H3"]["p"]) == "1e-6"
      and S1["H2"]["rejected"] and S1["H3"]["rejected"] and (S1["H2"]["wins"], S1["H2"]["losses"]) == (32, 47),
      "Table V S1 H2/H3: -0.002 (32 wins, 47 losses), p 1e-4 / 1e-6, supported")
check(all(S2[h]["n"] == 96 and S2[h]["p"] <= 0.002 and S2[h]["rejected"] for h in ("H1", "H2", "H3")), "Table V S2: n 96, p <= 0.002")
lg = conf["secondary_log1p"]["GW-HAC_minus_FL-U8_log1p"]
check(eq(lg["median_diff"], 0.0, 3) and eq(lg["ci_lo"], -0.001, 3) and eq(lg["ci_hi"], 0.002, 3), "Table V log(1+x): 0.000 [-0.001, +0.002]")

pdf = pd.read_csv(T / "confirm_per_dataset.csv").set_index("dataset")
meds = tuple(round(pdf[m].median(), 3) for m in ("GW-HAC", "FL-U8", "KM"))
check(meds == (0.040, 0.043, 0.039), f"median held-out ARI GW-HAC, FL-U8, k-means: {meds}")
unin = (pdf[["FL-U8", "GW-HAC", "GW-nw", "KM"]] < 0.02).all(axis=1)
check(unin.sum() == 38, f"uninformative datasets (every method below 0.02): {unin.sum()}")
diff = (pdf["GW-HAC"] - pdf["FL-U8"])[~unin]
check_print(round(diff.mean(), 3) == -0.019, abs(diff.mean() + 0.019) <= 0.001, "informative subset: mean GW-HAC - FL-U8, paper -0.019",
            f"{diff.mean():+.5f} -> {diff.mean():+.3f}")
for ds, gw, fl in (("1496_ringnorm", 0.005, 0.781), ("1511_wholesale_customers", -0.01, 0.58),
                   ("756_autoPrice", 0.67, 0.21), ("40705_tokyo1", 0.36, 0.10)):
    k = len(str(abs(gw)).split(".")[1]) if gw else 3
    kf = len(str(fl).split(".")[1])
    check(eq(pdf.loc[ds, "GW-HAC"], gw, k) and eq(pdf.loc[ds, "FL-U8"], fl, kf),
          f"Fig. 2 {ds}: GW-HAC {pdf.loc[ds, 'GW-HAC']:.3f}, FL-U8 {pdf.loc[ds, 'FL-U8']:.3f}")

if errata:
    print("\nErrata of the submitted manuscript (off by one unit in the last printed digit; correct in a revision):")
    for e in errata:
        print("  - " + e)
print(f"\n{n - len(fails)}/{n} TETCI checks passed ({len(errata)} with a printing erratum)")
sys.exit(1 if fails else 0)
