"""Bridge to the third-party MATLAB baselines run under GNU Octave (external/README.md, octave/*.m)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / "octave", ROOT / "external" / "OCL_Source_Code" / "OCL_Source_Code", ROOT / "external" / "IS-TSK-FC"]


def _run(expr: str, arrays: dict, timeout: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for k, a in arrays.items():
            np.savetxt(d / f"{k}.csv", np.asarray(a, float), delimiter=",", fmt="%.17g")
        add = "".join(f"addpath(genpath('{p}'));" for p in PATHS)
        loads = "".join(f"{k}=csvread('{d / (k + '.csv')}');" for k in arrays)
        code = (f"warning('off','all');pkg load fuzzy-logic-toolkit statistics;{add}{loads}"
                f"evalc('out={expr};');csvwrite('{d / 'out.csv'}', out(:));")
        r = subprocess.run(["octave", "--no-gui", "--norc", "-q", "--eval", code], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0 or not (d / "out.csv").exists():
            raise RuntimeError(f"octave failed: {r.stderr[-800:]}")
        return np.loadtxt(d / "out.csv", delimiter=",").astype(int).ravel()


def ocl(codes: np.ndarray, K: int, seed: int = 0, timeout: int = 3600) -> np.ndarray:
    """OCL partition (0-based labels) of integer codes; each column is re-indexed to 1..NumVal."""
    C = np.column_stack([np.unique(c, return_inverse=True)[1] + 1 for c in np.asarray(codes).T])
    return _run(f"ocl_labels(X,{int(K)},{int(seed)})", {"X": C}, timeout) - 1


def istsk(X: np.ndarray, init_labels: np.ndarray, R: int, lam: float, timeout: int = 3600) -> np.ndarray:
    """IS-TSK-FC partition (0-based labels relabelled to 0..K'-1) from initial labels.

    Constant columns are removed first: the authors' ``preproc`` drops them for the FCM antecedents while the rest of
    their code keeps the full matrix, which fails on such data; a constant feature carries no information."""
    X = np.asarray(X, float)
    X = X[:, X.std(0) > 0]
    y = _run(f"istsk_labels(X,Y,{int(R)},{float(lam)!r})", {"X": X, "Y": np.asarray(init_labels) + 1}, timeout)
    return np.unique(y, return_inverse=True)[1]
