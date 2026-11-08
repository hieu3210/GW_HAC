"""Margin delta and sample size n for the confirmatory study (pre-registration Section 4), from the development
factorial runs (raw/v2fac_<ds>.csv of the 25 development datasets).

    delta   = median over datasets of the across-split SD of FL-U8 ARI, rounded up to 0.005, floor 0.02
    d_i     = mean over splits of ARI_GW-HAC - ARI_FL-U8 on development dataset i (GW-HAC = HA-grid|native|interval, l=3)
    n       = smallest n on the grid 40, 50, ..., 200 whose simulated power is >= 0.80 for H1, where one simulated study
              draws n values with replacement from {d_i - median(d)} (true median 0), adds delta and applies the
              one-sided Wilcoxon signed-rank test at alpha_A / 3 = 0.025 / 3 (the smallest Holm threshold)
    n_max   = ceil(1.15 n)
The normal-approximation n = ceil(1.05 (z_.975 + z_.8)^2 sigma_d^2 / delta^2) is also reported (not used).

Output: <out>/tables/power.json

    python v2/70_power.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import v2lib as L


def sim_power(d0: np.ndarray, n: int, delta: float, alpha: float, R: int, rng) -> float:
    x = rng.choice(d0, (R, n), replace=True) + delta
    p = np.array([wilcoxon(r, zero_method="zsplit", alternative="greater").pvalue if np.any(r != 0) else 1.0 for r in x])
    return float(np.mean(p < alpha))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.RESULTS))
    a = ap.parse_args()
    raw = Path(a.out) / "raw"
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(raw / "v2fac_*.csv")))], ignore_index=True)
    df = df[df.group == "dev"]
    P = L.CFG["power"]
    fl = df[df.method == f"FL-U{L.CFG['fl_labels']}"]
    gw = df[(df.method == L.tag(L.PRIMARY)) & (df.level == L.MAIN_LEVEL)]
    delta_raw = float(fl.groupby("dataset").ari.std(ddof=1).median())
    step = P["delta_round"]
    delta = round(max(P["delta_floor"], math.ceil(round(delta_raw / step, 9)) * step), 4)
    d = (gw.set_index(["dataset", "split"]).ari - fl.set_index(["dataset", "split"]).ari).groupby("dataset").mean()
    sigma = float(d.std(ddof=1))
    n_formula = math.ceil(P["wilcoxon_are"] * (P["z_alpha"] + P["z_power"]) ** 2 * sigma ** 2 / delta ** 2)
    alpha = L.CFG["familyA_alpha"] / 3
    rng = np.random.default_rng(P["sim_seed"])
    d0 = d.to_numpy() - np.median(d)
    lo, hi, st = P["sim_n_grid"]
    curve, n_req = [], None
    for n in range(lo, hi + 1, st):
        pw = sim_power(d0, n, delta, alpha, P["sim_R"], rng)
        curve.append([n, pw])
        print(f"n={n:4d} power={pw:.3f}", flush=True)
        if pw >= P["sim_target"] and n_req is None:
            n_req = n
            break
    if n_req is None:
        n_req = hi
    res = dict(n_dev=int(d.notna().sum()), delta_raw=delta_raw, delta=delta, sigma_d=sigma,
               n_required=int(n_req), n_max=int(math.ceil(P["n_max_factor"] * n_req)), method=P["method"],
               sim_alpha=alpha, sim_R=P["sim_R"], sim_seed=P["sim_seed"], power_curve=curve,
               n_formula_not_used=int(n_formula), dev_median_diff_GWHAC_minus_FLU8=float(d.median()),
               datasets=sorted(d.index))
    (Path(a.out) / "tables").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "tables" / "power.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("datasets", "power_curve")}, indent=1))
    if res["n_dev"] != 25:
        print(f"WARNING: {res['n_dev']} development datasets found (expected 25).")


if __name__ == "__main__":
    main()
