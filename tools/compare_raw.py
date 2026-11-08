"""Compare recomputed per-split result files with the archive:  python tools/compare_raw.py <new/raw> <archive/raw>

Files must be byte-identical, with one documented exception: the column mse_lloyd_tr of v2feat_*.csv. The archived
v2feat files were written in the first development run with an iterative 1-D Lloyd quantizer; that reference was later
replaced by an exact dynamic-programming quantizer, and the price column used in the paper (Table 3) was recomputed by
v2/61b_price_fix.py into v2price_*.csv. Recomputing v2feat with the current code therefore changes mse_lloyd_tr only.
"""
import sys
from pathlib import Path

import pandas as pd

new, arch = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for f in sorted(new.glob("*.csv")):
    a = arch / f.name
    if f.read_bytes() == a.read_bytes():
        print(f"same      {f.name}")
    elif f.name.startswith("v2feat_"):
        x, y = pd.read_csv(f), pd.read_csv(a)
        same = x.drop(columns="mse_lloyd_tr").equals(y.drop(columns="mse_lloyd_tr"))
        print(f"{'same*' if same else 'DIFFERS':9s} {f.name}" + ("  (* all columns except mse_lloyd_tr, see docstring)" if same else ""))
        bad += not same
    else:
        print(f"DIFFERS   {f.name}")
        bad += 1
print("all recomputed files agree with the archive" if bad == 0 else f"{bad} file(s) differ")
sys.exit(1 if bad else 0)
