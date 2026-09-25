"""High-level identify -> derive -> estimate -> weight -> bootstrap workflows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import compute_weights, bootstrap, _positive_integer
from .density import validate_data
from .identification import identify, simplify_formula
from .symbolic import derive_weights, names


@dataclass(frozen=True)
class Analysis:
    transport_formula: object
    weight_expression: object

    @property
    def required_target_distributions(self):
        return self.weight_expression.required_target_distributions

    @property
    def required_source_distributions(self):
        return self.weight_expression.required_source_distributions

    def describe(self):
        plan = self.weight_expression
        return "\n".join([
            str(self.transport_formula), str(plan),
            "W_X = " + repr(plan.w_x), "W_R = " + repr(plan.w_r),
            "Target distributions: " + "; ".join(map(str, self.required_target_distributions)),
            "Source distributions: " + "; ".join(map(str, self.required_source_distributions)),
            "Kernel variables: " + repr(plan.kernel_variables),
        ])


def analyze(causal_graph=None, *, outcomes=None, interventions=None, selection=(),
            mode="standard", formula=None, simplify=False, info_print=False):
    """Identify a transport formula and derive its TCW expression.

    Alternatively pass formula=TransportFormula(...) from a verified
    derivation. Graph identification and manual formula input are exclusive.
    Both paths enforce Chapter 4's supported factorization condition.
    """
    if formula is not None:
        if simplify:
            raise ValueError("Graph simplification requires a graph query.")
        if causal_graph is not None or outcomes is not None or interventions is not None or selection:
            raise ValueError("Pass either a formula or a graph query, not both.")
    else:
        if causal_graph is None or outcomes is None or interventions is None:
            raise ValueError("Graph analysis requires causal_graph, outcomes and interventions.")
        formula = identify(causal_graph, outcomes, interventions, selection, mode=mode)
        if simplify:
            formula = simplify_formula(formula, causal_graph, selection)
    result = Analysis(formula, derive_weights(formula))
    if info_print:
        print(result.describe())
    return result


general_tcb_analysis = analyze


def observed_interventions(data, variables):
    """Return intervention dictionaries and observed counts (discrete labels).

    For continuous interventions supply a grid explicitly instead.
    """
    data = validate_data(data)
    variables = names(variables)
    if not variables:
        return [{}], [next(iter(data.values())).shape[0]]
    parts = [data[v].reshape(len(data[v]), -1) for v in variables]
    offsets = np.cumsum([0] + [a.shape[1] for a in parts])
    values, counts = np.unique(np.concatenate(parts, axis=1), axis=0, return_counts=True)
    result = [{v: row[offsets[i]:offsets[i+1]].copy() for i, v in enumerate(variables)} for row in values]
    return result, counts.tolist()


@dataclass(frozen=True)
class TCBResult:
    data: dict
    indices: np.ndarray
    intervention_ids: np.ndarray
    weight_results: tuple
    analysis: Analysis


def transportable_bootstrap(analysis, data, interventions, *, source_distributions,
                           target_distributions, sample_counts=None, n_samples=None,
                           random_state=None, shuffle=True, **weight_options):
    """Compute TCW and resample for each intervention (Chapter 4 Algorithm 5).

    n_samples is a TOTAL count, divided evenly unless sample_counts is
    explicitly supplied. Use observed_interventions() for class-count
    matching. Continuous intervention grids use the same interface.
    weight_options are passed to compute_weights, including coupling,
    kernels, integration_data and chunk sizes.
    """
    data = validate_data(data)
    interventions = list(interventions)
    if not interventions:
        raise ValueError("At least one intervention is required.")
    if sample_counts is not None and n_samples is not None:
        raise ValueError("Specify sample_counts or total n_samples, not both.")
    if sample_counts is None:
        total = next(iter(data.values())).shape[0] if n_samples is None else _positive_integer(n_samples, "n_samples")
        q, r = divmod(total, len(interventions))
        counts = [q + (i < r) for i in range(len(interventions))]
    else:
        counts = [_positive_integer(v, "sample_counts", allow_zero=True) for v in sample_counts]
        if len(counts) != len(interventions) or sum(counts) <= 0:
            raise ValueError("sample_counts must match interventions and have positive total.")
    rng = np.random.default_rng(random_state)
    parts, indices, labels, weights = [], [], [], []
    for i, (intervention, count) in enumerate(zip(interventions, counts)):
        result = compute_weights(
            analysis.weight_expression, data, intervention,
            source_distributions=source_distributions,
            target_distributions=target_distributions, **weight_options,
        )
        sampled = bootstrap(data, result, n_samples=count, random_state=rng)
        parts.append(sampled.data)
        indices.append(sampled.indices)
        labels.append(np.full(count, i, dtype=int))
        weights.append(result)
    combined = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    indices, labels = np.concatenate(indices), np.concatenate(labels)
    if shuffle:
        order = rng.permutation(len(indices))
        combined = {key: value[order] for key, value in combined.items()}
        indices, labels = indices[order], labels[order]
    return TCBResult(combined, indices, labels, tuple(weights), analysis)
