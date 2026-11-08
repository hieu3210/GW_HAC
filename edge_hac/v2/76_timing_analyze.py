"""Summary of 75_timing.py: per dataset, seconds mean seconds per stage over splits; then the median over
datasets of (a) seconds and (b) the ratio to k-means, per group.  Writes tables/timing_summary.csv and .tex.

    python v2/76_timing_analyze.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import v2lib as L

ORDER = ["calib_grid", "calib_theta", "semantize", "fit_gwhac", "GW-HAC total", "FL-U8", "KM", "GW-nw", "IMM+ExKMC"]


def main() -> None:
    a = L.common_args(__doc__).parse_args()
    out = Path(a.out)
    fs = sorted((out / "raw").glob("v2time_*.csv"))
    if not fs:
        raise SystemExit("no v2time_*.csv; run 75_timing.py first")
    d = pd.concat(map(pd.read_csv, fs))
    per = d.groupby(["group", "dataset", "N", "F", "K", "stage"]).seconds.mean().unstack("stage")
    per["GW-HAC total"] = per[["calib_grid", "semantize", "fit_gwhac"]].sum(axis=1)
    cols = [c for c in ORDER if c in per]
    rows = []
    for g, p in per.groupby(level="group"):
        for c in cols:
            rows.append(dict(group=g, stage=c, n=len(p), median_s=p[c].median(), max_s=p[c].max(),
                             median_ratio_to_KM=(p[c] / p["KM"]).median()))
    s = pd.DataFrame(rows)
    (out / "tables").mkdir(exist_ok=True)
    per.reset_index().to_csv(out / "tables" / "timing_per_dataset.csv", index=False)
    s.to_csv(out / "tables" / "timing_summary.csv", index=False)
    (out / "tables" / "timing_summary.tex").write_text(s.to_latex(index=False, float_format="%.3g"))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
