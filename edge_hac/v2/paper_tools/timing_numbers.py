"""Running-time numbers quoted in Section 7 ("Running time") and Section 8 of the paper.

Aggregates results/GW-HAC_v2/raw/v2time_*.csv (written by v2/75_timing.py) per dataset: mean over splits,
then the median over datasets.  Unlike v2/76_timing_analyze.py, the grouping key is (group, dataset) only.
76 also groups by the feature count F, which is computed after constant columns are removed and can differ
between splits, so 76 counts a few datasets twice (n = 28 / 133 instead of 25 / 126).

    python paper_tools/timing_numbers.py [results/archive]      (run from edge_hac/v2)
"""
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "results" / "archive")
d = pd.concat(map(pd.read_csv, sorted((R / "raw").glob("v2time_*.csv"))))
per = d.groupby(["group", "dataset", "stage"]).seconds.mean().unstack("stage")
NF = d.groupby(["group", "dataset"])[["N", "F"]].max()
per["GW-HAC total"] = per[["calib_grid", "semantize", "fit_gwhac"]].sum(axis=1)
per["GW-HAC (HA-theta)"] = per[["calib_theta", "semantize", "fit_gwhac"]].sum(axis=1)
per["calib share"] = per.calib_grid / per["GW-HAC total"]
per["fit share"] = (per.semantize + per.fit_gwhac) / per["GW-HAC total"]

rows = []
for g, p in per.groupby(level="group"):
    for c in ["calib_grid", "calib_theta", "semantize", "fit_gwhac", "GW-HAC total", "GW-HAC (HA-theta)",
              "FL-U8", "KM", "GW-nw", "IMM+ExKMC"]:
        rows.append(dict(group=g, stage=c, n=len(p), median_ms=1e3 * p[c].median(),
                         median_ratio_to_KM=(p[c] / p.KM).median(), max_s=p[c].max()))
s = pd.DataFrame(rows)
print(s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
for g, p in per.groupby(level="group"):
    print(f"\n[{g}] n={len(p)}  median calibration share {p['calib share'].median():.1%}, "
          f"fit share {p['fit share'].median():.1%}")
    worst = p["GW-HAC total"].idxmax()
    print(f"  slowest GW-HAC: {worst[1]} N={NF.loc[worst].N} F={NF.loc[worst].F} total {p.loc[worst, 'GW-HAC total']:.1f} s"
          f" (calibration {p.loc[worst, 'calib_grid']:.1f} s, k-means {p.loc[worst, 'KM']:.2f} s)")
    big = NF.loc[g].N.idxmax()
    print(f"  largest N: {big} N={NF.loc[(g, big)].N} F={NF.loc[(g, big)].F} GW-HAC {p.loc[(g, big), 'GW-HAC total']:.1f} s")
out = R / "tables" / "timing_summary_by_dataset.csv"
s.to_csv(out, index=False)
print(f"\nwrote {out}")
