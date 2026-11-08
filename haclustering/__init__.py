"""haclustering: interpretable clustering in the semantic space of Hedge Algebras.

This package provides:

- ``hedge_algebra``: a faithful implementation of a linear Hedge Algebra (HA),
  including the fuzziness measure ``fm``, the Semantically Quantifying Mapping
  (SQM) ``nu``, generation of the linguistic term set, and the semantization /
  desemantization (decoding) operations that turn numeric values into linguistic
  labels and back.

The clustering models and baselines are added on top of this core (see
``haclustering.clustering`` and ``haclustering.baselines``).
"""

from .clustering import HAClustering
from .isac_ha import ISACHAClustering
from .hedge_algebra import HedgeAlgebra, Term

__all__ = ["HedgeAlgebra", "Term", "HAClustering", "ISACHAClustering"]

__version__ = "0.1.0"
