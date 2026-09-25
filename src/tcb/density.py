"""Adapters to causalbootstrapping estimators and user-supplied distributions."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


def validate_data(data):
    """Accept a mapping or DataFrame; retain trailing dimensions for images."""
    if hasattr(data, "columns"):
        data = {str(k): data[k].to_numpy() for k in data.columns}
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a nonempty variable-to-array mapping or DataFrame.")
    arrays = {k: np.asarray(v) for k, v in data.items()}
    if any(not isinstance(k, str) or not k for k in arrays):
        raise ValueError("Data keys must be nonempty variable names.")
    if any(a.ndim == 0 for a in arrays.values()):
        raise ValueError("Each data variable needs a row axis.")
    sizes = {a.shape[0] for a in arrays.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("Data variables must have the same nonzero number of rows.")
    return arrays


def as_columns(array):
    array = np.asarray(array)
    return array.reshape(array.shape[0], -1)


def _stack(points, variables):
    result = np.concatenate([as_columns(points[v]) for v in variables], axis=1)
    if not np.isfinite(result.astype(float)).all():
        raise ValueError("Density input contains NaN or infinite values.")
    return result


@dataclass(frozen=True)
class DensityAdapter:
    """Adapt a fitted cb estimator that accepts an (N,d) matrix."""
    function: object
    variables: tuple

    def __call__(self, points):
        return self.function(_stack(points, self.variables))


def adapt_density(function, variables):
    """Column order is explicit and is never inferred from function names."""
    variables = (variables,) if isinstance(variables, str) else tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must give a nonempty, unique column order.")
    return DensityAdapter(function, variables)


@dataclass(frozen=True)
class CallableAdapter:
    function: object
    arguments: tuple
    vectorized: bool = True

    def __call__(self, points):
        args = [points[v] for v in self.arguments]
        if self.vectorized:
            return self.function(*args)
        values = []
        for row in zip(*args):
            result = np.asarray(self.function(*row), dtype=float)
            if result.size != 1:
                raise ValueError("A scalar distribution callback must return one value per row.")
            values.append(float(result.reshape(-1)[0]))
        return np.asarray(values)


def adapt_callable(function, arguments, *, vectorized=True):
    """Adapt functions such as p_z_doy_v(z, intv_y, v) from the experiments.

    arguments=("Z", "Y", "V") explicitly specifies positional order.
    vectorized=False supports the old length-one distribution functions.
    """
    arguments = (arguments,) if isinstance(arguments, str) else tuple(arguments)
    if not arguments or len(set(arguments)) != len(arguments):
        raise ValueError("arguments must contain unique variable names.")
    return CallableAdapter(function, arguments, vectorized)


def fit_target_distributions(expression, data, *, method="kde", bins=None, bandwidth=None):
    """Fit only the target joints that remain in the TCW expression.

    Reuses MultivarContiDistributionEstimator (KDE, histogram, multinorm).
    Histogram bins are specified per base variable: 0 means empirical
    category count, as in the existing experiments. The upstream histogram
    returns bin masses, not a continuous PDF; keep bins consistent across
    overlapping joints. Source interventional distributions are never fit.
    """
    from causalbootstrapping.distEst_lib import MultivarContiDistributionEstimator
    data = validate_data(data)
    if method not in ("kde", "histogram", "multinorm"):
        raise ValueError("method must be 'kde', 'histogram' or 'multinorm'.")
    fitted = {}
    for key in expression.required_target_distributions:
        missing = set(key.variables) - set(data)
        if missing:
            raise KeyError("Missing target density data: " + repr(sorted(missing)))
        matrix = _stack(data, key.variables)
        estimator = MultivarContiDistributionEstimator(matrix)
        if method == "kde":
            function = estimator.fit_kde(bandwidth=bandwidth)
        elif method == "multinorm":
            function = estimator.fit_multinorm()
        else:
            if bins is None:
                raise ValueError("histogram requires bins={variable: count} (0 for categories).")
            counts = []
            for variable in key.variables:
                if variable not in bins:
                    raise ValueError("Missing histogram bins for " + variable)
                dimensions = as_columns(data[variable]).shape[1]
                values = bins[variable]
                values = [values] * dimensions if np.isscalar(values) else list(values)
                if len(values) != dimensions or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
                    for v in values
                ):
                    raise ValueError("bins must give nonnegative integer counts for each column.")
                counts.extend(values)
            function = estimator.fit_histogram(n_bins=counts)
        fitted[key] = adapt_density(function, key.variables)
    return fitted
