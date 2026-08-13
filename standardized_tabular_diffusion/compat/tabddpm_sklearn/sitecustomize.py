"""Preserve TabDDPM's integral ``1e9`` subsample on modern scikit-learn.

Pinned TabDDPM passes ``subsample=1e9`` to ``QuantileTransformer``. Its
original scikit-learn 1.0.2 environment accepted that integral float, whereas
supported Python 3.11 releases validate the parameter as an integer. This
startup hook changes only integral float values to the exactly equal integer
before calling the unchanged official estimator. All other values and
arguments are forwarded without modification.
"""

from __future__ import annotations

import math
from typing import Any

import sklearn.preprocessing

_OfficialQuantileTransformer = sklearn.preprocessing.QuantileTransformer


class IntegralSubsampleQuantileTransformer(_OfficialQuantileTransformer):
    """API bridge with the exact official estimator signature."""

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int | float | None = 10000,
        random_state: Any = None,
        copy: bool = True,
    ) -> None:
        if isinstance(subsample, float):
            if not math.isfinite(subsample) or not subsample.is_integer():
                raise ValueError("TabDDPM compatibility accepts only integral float subsample values")
            subsample = int(subsample)
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )


sklearn.preprocessing.QuantileTransformer = IntegralSubsampleQuantileTransformer
