"""Flexible HA clustering variants for follow-up research.

This module provides a backward-compatible extension path on top of the current
EWA-HAC / FE-HAC core. The main design goal is to keep the legacy behavior
available while allowing research toggles for future work without mutating the
existing ``HAClustering`` implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .clustering import HAClustering
from .fe_hac import FEHAC
from .hedge_algebra import HedgeAlgebra


class ISACHAClustering:
    """Flexible HA clustering wrapper for research ablations.

    The wrapper preserves the current crisp EWA-HAC behavior by default, but it
    can delegate to FE-HAC when soft memberships or kernel distortion are
    requested. This keeps the research surface flexible without changing the
    original estimator contract.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        theta: float = 0.5,
        alpha: float = 0.5,
        entropy_gamma: Optional[float] = 1.0,
        sqm_lambda: float = 0.01,
        kernel: bool = False,
        fuzzy: bool = False,
        sigma: float = 0.5,
        t0_factor: float = 0.2,
        cooling: float = 0.9,
        t_min_ratio: float = 1e-3,
        inner_iter: int = 20,
        weighting: str = "entropy",
        init: str = "sqm",
        max_iter: int = 100,
        tol: float = 1e-5,
        decode_depth: int = 2,
        feature_names: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        ha: Optional[HedgeAlgebra] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.entropy_gamma = entropy_gamma
        self.sqm_lambda = sqm_lambda
        self.kernel = kernel
        self.fuzzy = fuzzy
        self.sigma = sigma
        self.t0_factor = t0_factor
        self.cooling = cooling
        self.t_min_ratio = t_min_ratio
        self.inner_iter = inner_iter
        self.weighting = weighting
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.decode_depth = decode_depth
        self.feature_names = feature_names
        self.random_state = random_state
        self.ha = ha if ha is not None else HedgeAlgebra(theta=theta, alpha=alpha)

    def _make_model(self):
        if self.kernel or self.fuzzy:
            return FEHAC(
                n_clusters=self.n_clusters,
                theta=self.theta,
                alpha=self.alpha,
                entropy_gamma=self.entropy_gamma,
                sqm_lambda=self.sqm_lambda,
                kernel=self.kernel,
                sigma=self.sigma,
                t0_factor=self.t0_factor,
                cooling=self.cooling,
                t_min_ratio=self.t_min_ratio,
                inner_iter=self.inner_iter,
                tol=self.tol,
                init=self.init,
                decode_depth=self.decode_depth,
                feature_names=self.feature_names,
                random_state=self.random_state,
                ha=self.ha,
            )

        return HAClustering(
            n_clusters=self.n_clusters,
            theta=self.theta,
            alpha=self.alpha,
            entropy_gamma=self.entropy_gamma,
            sqm_lambda=self.sqm_lambda,
            weighting=self.weighting,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            decode_depth=self.decode_depth,
            feature_names=self.feature_names,
            random_state=self.random_state,
            ha=self.ha,
        )

    def fit(self, X: np.ndarray) -> "ISACHAClustering":
        self._model = self._make_model().fit(X)
        self.labels_ = self._model.labels_
        self.cluster_centers_ = self._model.cluster_centers_
        self.feature_weights_ = getattr(self._model, "feature_weights_", None)
        self.cluster_descriptions_ = getattr(self._model, "cluster_descriptions_", None)
        self.objective_ = getattr(self._model, "objective_", None)
        self.J_history_ = getattr(self._model, "J_history_", None)
        self.F_history_ = getattr(self._model, "F_history_", None)
        self.memberships_ = getattr(self._model, "memberships_", None)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
