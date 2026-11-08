"""Compare regenerated outputs with the archive and with the numbers printed in the paper.

    python tools/check_outputs.py [out]          (default: out/)

1. Every file in out/tables that also exists in results/archive/tables must be byte-identical
   (tables of Sections 7.2-7.3, Tables 2, 3 and 5, power analysis, timing, Fig. 2 data).
2. The numerical verification of Section 6 (out/tables/numerics_sec6.json) must reproduce Table 1 and the in-text
   numbers of Section 6 to the printed precision.
3. The Iris example (out/worked_example_iris.txt) must reproduce Table 4.
Exit status 1 if any check fails.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out"
ARCH = ROOT / "results" / "archive"
fails, n = [], 0


def check(ok, msg):
    global n
    n += 1
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


# 1 -------------------------------------------------------------------------------------------------------------
SKIP = {"numerics_sec6.json", "tab1_medium.tex"}          # floating-point output of check 2, compared numerically
tabs = sorted(p for p in (OUT / "tables").glob("*") if p.name not in SKIP and (ARCH / "tables" / p.name).exists())
for p in tabs:
    check(p.read_bytes() == (ARCH / "tables" / p.name).read_bytes(), f"table identical to archive: {p.name}")
if not tabs:
    check(False, "no regenerated tables found in out/tables (run: sh reproduce.sh tables)")

# 2 -------------------------------------------------------------------------------------------------------------
js = OUT / "tables" / "numerics_sec6.json"
if js.exists():
    r = json.loads(js.read_text())
    T1 = {  # Table 1: median, theta*, grid (l=12), 1-4Phi*, 4Phi*/I3, 1/I3, recovered
        "Uniform": (0.500, 0.500, 0.500, 0.0, 1.000, 1.000, None),
        "Beta(2,5)": (0.264, 0.398, 0.400, 17.3, 1.481, 1.791, 39),
        "Beta(1,3)": (0.206, 0.379, 0.380, 19.4, 1.244, 1.543, 55),
        "Trunc. exp.": (0.115, 0.316, 0.315, 38.3, 1.270, 2.057, 74),
        "Mixture": (0.239, 0.355, 0.355, 14.2, 1.415, 1.649, 36)}
    for name, v in T1.items():
        d = r["densities"][name]
        got = (round(d["median"], 3), round(d["theta_star"], 3), round(d["grid_l12"], 3), round(100 * d["reduction"], 1),
               round(d["price_ha"], 3), round(d["price_dyadic"], 3),
               None if name == "Uniform" else round(100 * d["recovered"]))
        check(got == v, f"Table 1 row {name}: {got}")
        check(d["between"] and d["grid_within_step"], f"theta* between median and 1/2, grid within 0.005 ({name})")
        check(d["rel_err_limit_l12"] < 1e-4, f"4^12 D(theta*,1/2) vs Phi/3: rel. error {d['rel_err_limit_l12']:.1e} < 1e-4 ({name})")
    g = [v for dd in r["gradient_ratio"].values() for v in dd.values()]
    check(0.988 <= min(g) and max(g) <= 1.003, f"gradient ratio in [0.988, 1.003]: [{min(g):.4f}, {max(g):.4f}]")
    for name, want in (("Beta(2,5)", -0.070), ("Trunc. exp.", 0.035), ("Mixture", -0.023)):
        s = r["fig1b"][name]["scaled"][-1]
        check(abs(s - want) <= 0.0006, f"l=12 scaled optimum {name}: {s:+.4f} (paper {want:+.3f})")
    for name, want in (("Beta(2,5)", -0.065), ("Trunc. exp.", 0.035), ("Mixture", -0.022)):
        s = r["fig1b"][name]["predicted"][-1]
        check(abs(s - want) <= 0.0006, f"l=12 prediction {name}: {s:+.4f} (paper {want:+.3f})")
    g3 = max(v["3"] for v in r["alpha_gain"].values())
    g12 = max(v["12"] for v in r["alpha_gain"].values())
    check(g3 < 0.03 and g12 < 0.005, f"gain from tuning alpha: max {100*g3:.2f}% at l=3 (<3%), {100*g12:.2f}% at l=12 (<0.5%)")
    fa = r["fig1a"]
    i10 = fa["levels"].index(10)
    got = (round(fa["ha"][i10], 3), round(fa["dyadic"][i10], 3), round(fa["companding"][i10], 3))
    check(got == (1.269, 2.057, 1.000), f"price at l=10 (HA, dyadic, companding): {got}")
    th = dict(zip(fa["levels"], fa["theta_opt"]))
    check(round(th[2], 2) == 0.33 and all(abs(th[l] - 0.315) < 1e-9 for l in th if l >= 7),
          f"optimal theta: {th[2]:.3f} at l=2, {sorted({th[l] for l in th if l >= 7})} for l>=7")
    dec = r["decomposition"]
    check(max(x["identity_error"] for x in dec) < 1e-12, "Proposition 6 identity holds to machine precision (synthetic)")
else:
    check(False, "no out/tables/numerics_sec6.json (run: sh reproduce.sh numerics)")

# 3 -------------------------------------------------------------------------------------------------------------
we = OUT / "worked_example_iris.txt"
if we.exists():
    t = we.read_text()
    ari = re.findall(r"level (\d) ARI ([0-9.]+) agree ([0-9.]+)", t)
    check([(a, b) for a, b, _ in ari] == [("3", "0.741"), ("2", "0.681"), ("1", "0.605")], f"Table 4 ARI by level: {ari}")
    check([c for *_, c in ari][1:] == ["0.973", "0.76"], "projected descriptions reproduce 97% and 76% of assignments")
    words = re.findall(r"cluster \d size (\d+) lvl 3 (\[.*\])", t)
    check(words == [("21", "['LV-', 'VL+', 'VV-', 'VV-']"), ("38", "['VL+', 'LV-', 'VL-', 'VL+']"),
                    ("16", "['LV+', 'VL-', 'LV+', 'LV+']")], "Table 4 level-3 prototypes and cluster sizes")
else:
    check(False, "no out/worked_example_iris.txt (run: sh reproduce.sh tables)")

print(f"\n{n - len(fails)}/{n} checks passed")
sys.exit(1 if fails else 0)
