"""Dataset registry: the 18 tabular benchmarks of the EWA-HAC study + 7 edge/sensor datasets.

``load(name)`` returns ``X`` (float64, N x F), ``y`` (int, N) and optional ``batch`` (drift datasets).
``K`` for clustering is the number of classes, as in the EWA-HAC protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CORE_DIR = REPO / "data"                       # 18 CSVs shared by the repository (last column = target)
EDGE_DIR = REPO / "edge_hac" / "data" / "edge"

UCI18 = ("iris", "seeds", "wine", "glass", "ecoli", "ionosphere", "vehicle", "breast_cancer", "segment",
         "waveform", "thyroid", "banknote", "pima", "yeast", "vowel", "parkinsons", "sonar", "libras")
EDGE7 = ("har", "gas_drift", "beed_eeg", "pendigits", "satimage", "optdigits", "eeg_eye_state")
ALL = UCI18 + EDGE7


def load(name: str) -> Dict[str, np.ndarray]:
    if name in UCI18:
        df = pd.read_csv(CORE_DIR / f"{name}.csv")
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy()
        out = {"X": X, "y": pd.factorize(y)[0].astype(np.int64)}
    elif name in EDGE7:
        z = np.load(EDGE_DIR / f"{name}.npz")
        out = {k: z[k] for k in z.files}
        out["X"] = out["X"].astype(float)
    else:
        raise KeyError(name)
    keep = np.isfinite(out["X"]).all(axis=1)
    if not keep.all():
        out = {k: v[keep] for k, v in out.items()}
    return out


def available() -> Tuple[str, ...]:
    return tuple(n for n in ALL if (n in UCI18 and (CORE_DIR / f"{n}.csv").exists())
                 or (n in EDGE7 and (EDGE_DIR / f"{n}.npz").exists()))


def split(n: int, seed: int, frac: float = 0.5):
    """Random train/test index split (deterministic)."""
    perm = np.random.default_rng(1000 + seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[:k]), np.sort(perm[k:])
