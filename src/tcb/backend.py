"""Numerical TCW evaluation and index-based bootstrap resampling."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import logsumexp

from .density import as_columns, validate_data
from .exceptions import ApproximationWarning, MissingDistributionError, PositivityError


def delta_kernel(observed, intervention):
    """Kronecker kernel for discrete intervention variables, including vectors."""
    return np.all(observed == intervention, axis=1).astype(float)


def gaussian_kernel(bandwidth):
    """Explicit continuous intervention kernel; bandwidth must be positive."""
    bandwidth = np.asarray(bandwidth, dtype=float)
    if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0) or bandwidth.ndim > 1:
        raise ValueError("bandwidth must be a positive scalar or vector.")

    def kernel(observed, intervention):
        scaled = (observed - intervention) / bandwidth
        return np.exp(-0.5 * np.sum(scaled**2, axis=1)) / np.prod(
            np.broadcast_to(bandwidth, (observed.shape[1],)) * np.sqrt(2 * np.pi)
        )
    return kernel


def _positive_integer(value, label, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label + " must be an integer.")
    if value < (0 if allow_zero else 1):
        raise ValueError(label + (" must be nonnegative." if allow_zero else " must be positive."))
    return int(value)


def _intervention_values(intervention, expected=None):
    if expected is not None and set(intervention) != set(expected):
        raise ValueError("intervention must specify exactly " + repr(tuple(expected)))
    result = {}
    for key, value in intervention.items():
        array = np.asarray(value)
        if array.ndim > 1 or array.size == 0:
            raise ValueError("Each intervention must be a scalar or a fixed one-dimensional vector.")
        result[key] = array.reshape(-1).copy()
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Intervention values must be finite.")
    return result


def _evaluated_vector(value, count, label):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array))
    elif array.shape == (count, 1):
        array = array[:, 0]
    elif array.shape != (count,):
        raise ValueError(f"{label} must return a scalar, ({count},), or ({count},1); got {array.shape}.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise PositivityError(label + " returned negative or non-finite values.")
    return array


@dataclass(frozen=True)
class WeightResult:
    """One weight per original row and diagnostics for this intervention."""
    weights: np.ndarray
    log_weights: np.ndarray
    probabilities: np.ndarray
    intervention: dict
    coupling: str
    integration_size: int
    normalized: bool

    @property
    def effective_sample_size(self):
        return float(1.0 / np.dot(self.probabilities, self.probabilities))

    @property
    def raw_weights(self):
        with np.errstate(over="raise"):
            return np.exp(self.log_weights)


def compute_weights(expression, data, intervention, *, source_distributions,
                    target_distributions, kernels=None, coupling="product",
                    integration_data=None, batch_size=128, integration_batch_size=256,
                    normalize=True, compress_integration=True):
    """Evaluate Chapter 4 TCW using the empirical product carrier by default.

    source_distributions is mandatory, even when {} suffices. Each registry
    maps Probability keys to callbacks fn(points)->one probability/density
    per row; point values are (B,d) arrays keyed by base variable name.

    product: w_n = sum_j K*Psi(n,j)/(N*M), with M=N by default.
    diagonal: w_n = K*Psi(n,n)/N, reproducing the older experiment scripts.
    No NxM matrix is retained. Exact duplicate integration rows are grouped
    with their multiplicities; this changes neither the sum nor its scale.
    """
    data = validate_data(data)
    n = next(iter(data.values())).shape[0]
    batch_size = _positive_integer(batch_size, "batch_size")
    integration_batch_size = _positive_integer(integration_batch_size, "integration_batch_size")
    if coupling not in ("product", "diagonal"):
        raise ValueError("coupling must be 'product' or 'diagonal'.")
    intv = _intervention_values(intervention, expression.formula.interventions)
    kernels = {} if kernels is None else kernels
    for key in expression.kernel_variables:
        if key not in kernels or not callable(kernels[key]):
            raise ValueError("Provide kernels['" + key + "'] (delta_kernel for discrete, gaussian_kernel for continuous).")
    for domain, needed, registry in (
        ("source", expression.required_source_distributions, source_distributions),
        ("target", expression.required_target_distributions, target_distributions),
    ):
        if registry is None:
            raise MissingDistributionError(domain + "_distributions must be an explicit mapping.")
        absent = [str(key) for key in needed if key not in registry or not callable(registry[key])]
        if absent:
            raise MissingDistributionError("Missing " + domain + " distributions: " + "; ".join(absent))
    aliases = expression.formula.aliases
    base = lambda v: aliases.get(v, v)
    left_names = {base(v) for v in expression.w_x if v not in intv} | set(expression.kernel_variables)
    if left_names - set(data):
        raise KeyError("Missing observed data: " + repr(sorted(left_names - set(data))))
    for variable in set(intv) & set(data):
        if as_columns(data[variable]).shape[1] != len(intv[variable]):
            raise ValueError("Intervention dimension does not match data for " + variable)
    if coupling == "diagonal" and integration_data is not None:
        raise ValueError("diagonal coupling cannot use a separate integration dataset.")
    right = data if integration_data is None else validate_data(integration_data)
    right_names = sorted({base(v) for v in expression.w_r})
    if set(right_names) - set(right):
        raise KeyError("Missing integration data: " + repr(sorted(set(right_names) - set(right))))
    m = next(iter(right.values())).shape[0] if expression.w_r else 1
    right_indices = np.arange(m)
    multiplicities = np.ones(m, dtype=int)
    if expression.w_r and coupling == "diagonal":
        warnings.warn(
            "diagonal coupling uses same-row particles. It is the older experiment "
            "approximation, not Chapter 4's full empirical-product TCW.",
            ApproximationWarning, stacklevel=2,
        )
    elif expression.w_r and compress_integration:
        joint = np.concatenate([as_columns(right[v]) for v in right_names], axis=1)
        if np.issubdtype(joint.dtype, np.number):
            if not np.isfinite(joint).all():
                raise ValueError("Integration data contains non-finite values.")
            _, right_indices, multiplicities = np.unique(joint, axis=0, return_index=True, return_counts=True)

    def log_psi(rows, integration_rows):
        count = len(rows)
        # Evaluate the kernel first. A zero kernel excludes a row from the
        # relevant support, so off-support density ratios need not be evaluated.
        kernel = np.ones(count)
        for v in expression.kernel_variables:
            observed = as_columns(data[v][rows])
            assigned = np.broadcast_to(intv[v], observed.shape)
            kernel *= _evaluated_vector(kernels[v](observed, assigned), count, "Kernel for " + v)
        active = kernel > 0
        result = np.full(count, -np.inf)
        if not np.any(active):
            return result
        selected_rows = rows[active]
        selected_right = integration_rows[active]
        log_value = np.log(kernel[active])
        cache = {}
        for sign, factors in ((1, expression.numerator), (-1, expression.denominator)):
            for factor in factors:
                if factor not in cache:
                    key = factor.mapped(aliases)
                    points = {}
                    for symbol in factor.symbols:
                        variable = base(symbol)
                        if symbol in intv:
                            value = np.broadcast_to(intv[symbol], (len(selected_rows), len(intv[symbol])))
                        elif symbol in expression.w_r:
                            value = as_columns(right[variable][selected_right])
                        else:
                            value = as_columns(data[variable][selected_rows])
                        points[variable] = value
                    registry = source_distributions if factor.domain == "source" else target_distributions
                    cache[factor] = _evaluated_vector(registry[key](points), len(selected_rows), str(key))
                values = cache[factor]
                if sign < 0 and np.any(values <= 0):
                    raise PositivityError(
                        f"Non-positive denominator {factor} on {int(np.sum(values <= 0))} evaluated points. "
                        "Check overlap and density estimation; no clipping or replacement was applied."
                    )
                with np.errstate(divide="ignore"):
                    log_value = log_value + sign * np.log(values)
        result[active] = log_value
        return result

    logs = np.full(n, -np.inf)
    for start in range(0, n, batch_size):
        rows = np.arange(start, min(n, start + batch_size))
        if not expression.w_r or coupling == "diagonal":
            logs[rows] = log_psi(rows, rows if expression.w_r else np.zeros(len(rows), dtype=int)) - np.log(n)
        else:
            subtotal = np.full(len(rows), -np.inf)
            for pos in range(0, len(right_indices), integration_batch_size):
                ids = right_indices[pos:pos + integration_batch_size]
                counts = multiplicities[pos:pos + integration_batch_size]
                values = log_psi(np.repeat(rows, len(ids)), np.tile(ids, len(rows))).reshape(len(rows), len(ids))
                subtotal = np.logaddexp(subtotal, logsumexp(values + np.log(counts)[None, :], axis=1))
            logs[rows] = subtotal - np.log(n) - np.log(m)
    total = logsumexp(logs)
    if not np.isfinite(total):
        raise PositivityError("All TCW weights are zero or invalid for this intervention.")
    probabilities = np.exp(logs - total)
    probabilities /= probabilities.sum()
    if normalize:
        weights = probabilities.copy()
    else:
        with np.errstate(over="raise"):
            weights = np.exp(logs)
        if not np.any(weights > 0):
            raise PositivityError("Raw weights underflowed; use normalize=True and inspect log_weights.")
    return WeightResult(weights, logs, probabilities, intv, coupling, m, bool(normalize))


def weight_compute(*args, **kwargs):
    """Array-only lower-level interface, analogous to causalbootstrapping."""
    return compute_weights(*args, **kwargs).weights


@dataclass(frozen=True)
class BootstrapResult:
    data: dict
    indices: np.ndarray


def bootstrap(data, weights, *, intervention=None, n_samples=None, random_state=None):
    """Resample aligned arrays and append intv_<name>; preserve original labels.

    The upstream fast sampler draws indices only. Indexing original arrays
    here avoids its prime-name stripping and preserves image dimensions.
    """
    from causalbootstrapping.backend import cw_bootstrapper
    data = validate_data(data)
    n = next(iter(data.values())).shape[0]
    count = n if n_samples is None else _positive_integer(n_samples, "n_samples", allow_zero=True)
    if isinstance(weights, WeightResult):
        probabilities = weights.probabilities
        if intervention is None:
            intervention = weights.intervention
        else:
            checked = _intervention_values(intervention, weights.intervention)
            if any(not np.array_equal(checked[k], weights.intervention[k]) for k in checked):
                raise ValueError("Bootstrap labels must match the intervention used to compute weights.")
    else:
        probabilities = np.asarray(weights, dtype=float)
        if probabilities.shape == (n, 1):
            probabilities = probabilities[:, 0]
        if probabilities.shape != (n,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("weights must have shape (N,) or (N,1), be finite and nonnegative.")
        scale = probabilities.max()
        if scale <= 0:
            raise PositivityError("At least one bootstrap weight must be positive.")
        probabilities = probabilities / scale
        probabilities = probabilities / probabilities.sum()
    if probabilities.shape != (n,):
        raise ValueError("Weights and data row counts differ.")
    intv = _intervention_values({} if intervention is None else intervention)
    if any("intv_" + k in data for k in intv):
        raise ValueError("An intv_ output label would overwrite an existing data variable.")
    if isinstance(random_state, np.random.Generator):
        random_state = int(random_state.integers(0, 2**32 - 1))
    sampled = cw_bootstrapper(
        data={"row_index": np.arange(n)}, weights=probabilities, intv_dict={},
        n_sample=count, sampling_mode="fast", random_state=random_state,
    )
    indices = np.asarray(sampled["row_index"], dtype=int)
    output = {key: value[indices] for key, value in data.items()}
    output.update({"intv_" + key: np.broadcast_to(value, (count, len(value))).copy() for key, value in intv.items()})
    return BootstrapResult(output, indices)
