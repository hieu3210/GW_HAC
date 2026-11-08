"""Dataset loading for the clustering benchmark.

Datasets are read from local CSV files in ``data/`` (last column = integer
``target``). Use :func:`export_datasets` to (re)materialise them from scikit-learn
and OpenML. Reading from cached CSVs keeps experiments reproducible and offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Canonical benchmark suite (name -> expected #clusters from ground truth).
BENCHMARK = {
    "iris": 3,
    "seeds": 3,
    "wine": 3,
    "glass": 6,
    "ecoli": 8,
    "ionosphere": 2,
    "vehicle": 4,
    "breast_cancer": 2,
    # Extended suite: larger N and F to address reviewer M6 critique.
    "segment": 7,       # N=2310, F=19 (Image Segmentation)
    "waveform": 3,      # N=5000, F=40 (Waveform-5000; F > 34 max of original suite)
    # R1 extension (pre-registered in results/metrics/r1/dataset_selection.json).
    "thyroid": 3,       # N=215,  F=5
    "banknote": 2,      # N=1372, F=4
    "pima": 2,          # N=768,  F=8
    "yeast": 10,        # N=1484, F=8
    "vowel": 11,        # N=990,  F=10 (numeric attributes only)
    "parkinsons": 2,    # N=195,  F=22
    "sonar": 2,         # N=208,  F=60
    "libras": 15,       # N=360,  F=90
}

# The ten datasets of the original (S1) submission.
ORIGINAL_SUITE = [
    "iris", "seeds", "wine", "glass", "ecoli", "ionosphere", "vehicle",
    "breast_cancer", "segment", "waveform",
]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]

    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.y))

    @property
    def desc(self) -> str:
        return f"(N={self.X.shape[0]}, F={self.X.shape[1]}, K={self.n_clusters})"


def load(name: str) -> Dataset:
    """Load a single dataset by name from ``data/<name>.csv``."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m haclustering.datasets` to export datasets."
        )
    df = pd.read_csv(path)
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    feature_names = [c for c in df.columns if c != "target"]
    # Drop rows with NaNs (some OpenML sets have a few).
    mask = np.isfinite(X).all(axis=1)
    return Dataset(name, X[mask], y[mask], feature_names)


def load_benchmark(names: Optional[List[str]] = None) -> Dict[str, Dataset]:
    """Load the benchmark datasets (all by default), skipping any missing CSVs."""
    out: Dict[str, Dataset] = {}
    for name in (names if names is not None else BENCHMARK):
        try:
            out[name] = load(name)
        except FileNotFoundError:
            print(f"[datasets] skipping {name} (CSV not found)")
    return out


def export_datasets() -> None:
    """Materialise the benchmark CSVs from scikit-learn and OpenML."""
    from sklearn.datasets import (
        load_breast_cancer,
        load_iris,
        load_wine,
        fetch_openml,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name, X, y, names):
        df = pd.DataFrame(X, columns=list(names))
        _, y = np.unique(np.asarray(y).astype(str), return_inverse=True)
        df["target"] = y
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"[datasets] saved {name}.csv {df.shape}")

    for name, ld in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        d = ld()
        _save(name, d.data, d.target, d.feature_names)

    # OpenML name -> local CSV name (same when equal, else explicit mapping).
    openml_map = {
        "seeds": "seeds",
        "glass": "glass",
        "ecoli": "ecoli",
        "vehicle": "vehicle",
        "ionosphere": "ionosphere",
        "segment": "segment",        # N=2310, F=19, K=7 (Image Segmentation)
        "waveform-5000": "waveform",  # N=5000, F=40, K=3
        # R1 extension.
        "thyroid-new": "thyroid",
        "banknote-authentication": "banknote",
        "diabetes": "pima",
        "yeast": "yeast",
        "vowel": "vowel",
        "parkinsons": "parkinsons",
        "sonar": "sonar",
        "libras_move": "libras",
    }
    for openml_name, local_name in openml_map.items():
        try:
            d = fetch_openml(name=openml_name, version=1, as_frame=True, parser="auto")
            frame = d.frame
            ycol = d.target.name if getattr(d, "target", None) is not None else frame.columns[-1]
            y = frame[ycol]
            feats = frame.drop(columns=[ycol]).select_dtypes(include=[np.number])
            _save(local_name, feats.to_numpy(float), y, feats.columns)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[datasets] could not fetch {openml_name}: {exc!r}")


if __name__ == "__main__":
    export_datasets()
