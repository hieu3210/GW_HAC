"""Clustering evaluation metrics (external + internal)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cluster purity: fraction of points in the majority class of each cluster."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        members = y_true[y_pred == c]
        if members.size:
            total += np.bincount(members).max() if members.min() >= 0 else 0
    return total / len(y_true)


def evaluate(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute the full metric panel for one clustering result.

    ``X`` is the (commonly preprocessed) feature matrix used for internal indices.
    Internal indices need at least 2 distinct clusters.
    """
    n_labels = len(np.unique(labels))
    valid_internal = 1 < n_labels < len(X)
    return {
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "Purity": purity_score(y_true, labels),
        "Silhouette": silhouette_score(X, labels) if valid_internal else float("nan"),
        "DaviesBouldin": davies_bouldin_score(X, labels) if valid_internal else float("nan"),
    }
