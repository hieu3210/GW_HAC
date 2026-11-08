"""Faithful implementation of a linear Hedge Algebra (HA).

This module implements the algebraic machinery of Nguyen Cat Ho & Wechler's
Hedge Algebra theory in a form that is directly usable for data analysis:

* the **fuzziness measure** ``fm``,
* the **Semantically Quantifying Mapping** (SQM) ``nu`` that maps every
  linguistic term to a point in ``[0, 1]``,
* generation of the **linguistic term set** up to a given hedge depth,
* **semantization** (numeric value -> semantic value in ``[0, 1]``) and
  **desemantization / decoding** (semantic value -> nearest linguistic term),
  i.e. the operations that give clusters human-readable linguistic labels.

Construction of the SQM
-----------------------
A linear HA for one linguistic variable is ``AX = (X, G, C, H, <=)`` with

* generators ``G = {c-, c+}`` (e.g. *Low* < *High*),
* constants ``C = {0, W, 1}`` (absolutely-low, neutral, absolutely-high),
* hedges ``H = H- u H+`` (negative hedges such as *Little*; positive hedges
  such as *Very*),
* a semantic order ``<=``.

The fuzziness measure satisfies ``fm(c-) + fm(c+) = 1``,
``fm(h x) = mu(h) * fm(x)`` and ``sum_{h in H-} mu(h) = alpha``,
``sum_{h in H+} mu(h) = beta`` with ``alpha + beta = 1``.

We materialise the SQM through the equivalent and numerically robust
**fuzziness-interval construction**: every term ``x`` owns an interval
``I(x) subset [0, 1]`` of length ``fm(x)``; the hedges partition ``I(x)`` into
sub-intervals (one per hedge) whose order along the real line is governed by the
sign of ``x``. The SQM value ``nu(x)`` is the boundary, inside ``I(x)``, between
the negative- and positive-hedge groups. This reproduces the classical
closed-form values, e.g. with ``theta = fm(c-)``::

    nu(c-) = theta * (1 - alpha)
    nu(W)  = theta
    nu(c+) = theta + alpha * (1 - theta)

and extends consistently to arbitrarily long hedge strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Term:
    """A canonical linguistic term ``h_k ... h_1 c``.

    Attributes
    ----------
    generator:
        ``"-"`` for ``c-`` or ``"+"`` for ``c+``.
    hedges:
        Hedge keys applied from innermost to outermost. ``("L", "V")`` means
        ``V(L(c))`` -- first ``L`` is applied to the generator, then ``V``.
    """

    generator: str
    hedges: Tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.hedges)


@dataclass
class HedgeAlgebra:
    """A linear Hedge Algebra for a single linguistic variable.

    Parameters
    ----------
    theta:
        Fuzziness measure of the negative generator, ``fm(c-) = theta`` (so
        ``fm(c+) = 1 - theta``). Must lie in ``(0, 1)``.
    alpha:
        Total fuzziness measure of the negative hedges,
        ``sum_{h in H-} mu(h) = alpha`` (so the positive hedges share
        ``beta = 1 - alpha``). Must lie in ``(0, 1)``.
    neg_hedges, pos_hedges:
        Ordered hedge keys for ``H-`` and ``H+``. The default models the
        canonical two-hedge algebra ``H- = {L}`` (*Little*), ``H+ = {V}``
        (*Very*).
    hedge_mu:
        Optional explicit fuzziness measure per hedge. When omitted, ``alpha``
        is split uniformly across ``neg_hedges`` and ``beta`` across
        ``pos_hedges``.
    names:
        Human-readable labels for generators and hedges, used when decoding a
        semantic value into a phrase such as ``"Very High"``.
    """

    theta: float = 0.5
    alpha: float = 0.5
    neg_hedges: Sequence[str] = ("L",)
    pos_hedges: Sequence[str] = ("V",)
    hedge_mu: Optional[Dict[str, float]] = None
    names: Dict[str, str] = field(
        default_factory=lambda: {
            "-": "Low",
            "+": "High",
            "W": "Medium",
            "L": "Little",
            "V": "Very",
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.theta < 1.0:
            raise ValueError("theta must be in (0, 1)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not self.neg_hedges or not self.pos_hedges:
            raise ValueError("at least one negative and one positive hedge required")

        self.neg_hedges = tuple(self.neg_hedges)
        self.pos_hedges = tuple(self.pos_hedges)
        self.beta = 1.0 - self.alpha

        # Fuzziness measure of each hedge.
        if self.hedge_mu is None:
            mu: Dict[str, float] = {}
            for h in self.neg_hedges:
                mu[h] = self.alpha / len(self.neg_hedges)
            for h in self.pos_hedges:
                mu[h] = self.beta / len(self.pos_hedges)
            self.hedge_mu = mu
        else:
            neg_sum = sum(self.hedge_mu[h] for h in self.neg_hedges)
            pos_sum = sum(self.hedge_mu[h] for h in self.pos_hedges)
            if not np.isclose(neg_sum, self.alpha) or not np.isclose(pos_sum, self.beta):
                raise ValueError(
                    "hedge_mu must sum to alpha over H- and to beta over H+"
                )

        # Linear order of hedges along the semantic axis, used to lay out the
        # sub-intervals of a term. Negative hedges precede positive hedges.
        self._hedge_order: Tuple[str, ...] = self.neg_hedges + self.pos_hedges
        self._hedge_sign: Dict[str, int] = {
            **{h: -1 for h in self.neg_hedges},
            **{h: +1 for h in self.pos_hedges},
        }

    # ------------------------------------------------------------------ #
    # Core HA quantities: sign, fuzziness measure, fuzziness interval, SQM
    # ------------------------------------------------------------------ #
    def sign(self, term: Term) -> int:
        """Sign ``Sgn(x) in {-1, +1}`` of a term (its side relative to ``W``)."""
        s = -1 if term.generator == "-" else +1
        for h in term.hedges:
            s *= self._hedge_sign[h]
        return s

    def fm(self, term: Term) -> float:
        """Fuzziness measure ``fm(x) = mu(h_k) ... mu(h_1) fm(c)``."""
        f = self.theta if term.generator == "-" else (1.0 - self.theta)
        for h in term.hedges:
            f *= self.hedge_mu[h]
        return f

    def interval(self, term: Term) -> Tuple[float, float]:
        """Fuzziness interval ``I(x) = [a, b]`` with ``b - a = fm(x)``."""
        # Generator intervals partition [0, 1].
        if term.generator == "-":
            lo, hi, s = 0.0, self.theta, -1
        else:
            lo, hi, s = self.theta, 1.0, +1

        for h in term.hedges:
            lo, hi = self._sub_interval(h, lo, hi, s)
            s *= self._hedge_sign[h]
        return lo, hi

    def nu(self, term: Term) -> float:
        """Semantically Quantifying Mapping ``nu(x) in [0, 1]``.

        ``nu(x)`` is the boundary inside ``I(x)`` separating the negative-hedge
        group (total measure ``alpha``) from the positive-hedge group (``beta``),
        with the side chosen according to ``Sgn(x)``.
        """
        lo, hi = self.interval(term)
        s = self.sign(term)
        f = hi - lo
        # Sgn = +1: negative group occupies the left part (length alpha * f).
        # Sgn = -1: orientation reverses, positive group on the left (beta * f).
        return lo + (self.alpha if s > 0 else self.beta) * f

    def _sub_interval(
        self, hedge: str, lo: float, hi: float, parent_sign: int
    ) -> Tuple[float, float]:
        """Sub-interval that ``hedge`` carves out of parent interval ``[lo, hi]``."""
        f = hi - lo
        order = self._hedge_order if parent_sign > 0 else self._hedge_order[::-1]
        offset = 0.0
        for h in order:
            width = self.hedge_mu[h] * f
            if h == hedge:
                return lo + offset, lo + offset + width
            offset += width
        raise KeyError(f"unknown hedge {hedge!r}")

    # ------------------------------------------------------------------ #
    # Linguistic term set
    # ------------------------------------------------------------------ #
    def generate_terms(self, max_depth: int = 1) -> List[Term]:
        """All terms with hedge depth ``<= max_depth``, sorted by ``nu``.

        ``max_depth = 0`` yields the two generators; ``max_depth = 1`` adds one
        hedge (e.g. *Very Low*, *Little High*); and so on. Larger depths give a
        finer, non-uniform partition of ``[0, 1]`` that is denser near the
        generators -- the granularity knob of the algebra.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        terms: List[Term] = [Term("-"), Term("+")]
        frontier = list(terms)
        for _ in range(max_depth):
            nxt: List[Term] = []
            for t in frontier:
                for h in self._hedge_order:
                    nxt.append(Term(t.generator, t.hedges + (h,)))
            terms.extend(nxt)
            frontier = nxt
        return sorted(terms, key=self.nu)

    # ------------------------------------------------------------------ #
    # Semantization / desemantization (decoding)
    # ------------------------------------------------------------------ #
    @staticmethod
    def semantize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Map real values from ``[lo, hi]`` linearly onto the semantic axis ``[0, 1]``."""
        values = np.asarray(values, dtype=float)
        span = hi - lo
        span = span if span != 0 else 1.0
        return np.clip((values - lo) / span, 0.0, 1.0)

    def label(self, semantic_value: float, max_depth: int = 1) -> str:
        """Decode a semantic value in ``[0, 1]`` into a linguistic phrase.

        Returns the phrase (e.g. ``"Very High"``) of the term whose ``nu`` is
        closest to ``semantic_value`` among terms up to ``max_depth``.
        """
        term = self.nearest_term(semantic_value, max_depth=max_depth)
        return self.term_phrase(term)

    def sqm_values(self, max_depth: int = 2) -> np.ndarray:
        """Sorted array of SQM values ν(t) for all terms up to ``max_depth``."""
        terms = self.generate_terms(max_depth)
        return np.array([self.nu(t) for t in terms])

    def nearest_term(self, semantic_value: float, max_depth: int = 1) -> Term:
        """Term up to ``max_depth`` whose ``nu`` is closest to ``semantic_value``."""
        terms = self.generate_terms(max_depth)
        nus = np.array([self.nu(t) for t in terms])
        return terms[int(np.argmin(np.abs(nus - semantic_value)))]

    def term_phrase(self, term: Term) -> str:
        """Human-readable phrase for a term, e.g. ``Term('+', ('L', 'V')) -> 'Very Little High'``."""
        words = [self.names.get(h, h) for h in reversed(term.hedges)]
        words.append(self.names.get(term.generator, term.generator))
        return " ".join(words)
