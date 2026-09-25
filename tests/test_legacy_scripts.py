"""Compare against actual supplied experiment functions, without running training.

These additional local integration tests skip if the reference scripts are
not beside this project. The portable unit tests independently check the math.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from tcb import (analyze, compute_weights, adapt_density, adapt_callable,
                 Probability as P, ApproximationWarning)


EXPERIMENTS = Path(__file__).resolve().parents[2] / 'source_scripts' / 'experiment'


def load_numerics(path, requested):
    if not path.exists():
        pytest.skip('Original experiment script is not present: ' + str(path))
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in requested]
    prelude = ast.parse('from __future__ import annotations\n').body
    namespace = {'np': np, 'dataclass': dataclass, 'norm': norm, '__name__': __name__}
    exec(compile(ast.Module(body=prelude + nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def matrices():
    rng = np.random.default_rng(47)
    data = {'Y': rng.integers(0, 2, (19, 1)), 'Z': rng.integers(0, 2, (19, 1)),
            'W': rng.normal(size=(19, 1)), 'X': rng.normal(size=(19, 2))}
    # Positive distributions with consistent marginalization and column order.
    def yz(points):
        y, z = np.column_stack(points).T if isinstance(points, list) else np.asarray(points).T
        return np.where(y == z, .35, .15)
    def z(points):
        return np.full(np.asarray(points).reshape(-1, 1).shape[0], .5)
    def wyz(points):
        array = np.column_stack(points) if isinstance(points, list) else np.asarray(points)
        w, y, zv = array.T
        return norm.pdf(w, loc=.4*y + .8*zv) * yz(np.column_stack([y, zv]))
    return data, wyz, yz, z


@pytest.mark.parametrize('relative, scalar_name', [
    ('syn_classification/run_nonident_tcb_30runs.py', '_as_probability_scalar'),
    ('semi_syn_bgMNIST/bgMNIST_tcb_30runs.py', '_as_probability'),
])
def test_historical_classification_and_bgmnist(relative, scalar_name):
    legacy = load_numerics(EXPERIMENTS / relative,
                           {'TCBConfig', '_sigmoid', '_bernoulli_p_do_y', '_p_z_do_y', scalar_name, 'tcb_weights'})
    graph = 'Y; Z; W; X; S; Y -> Z; Z -> W; W -> X; S -> X; Y <-> X; Y <-> Z;'
    analysis = analyze(graph, outcomes='X', interventions='Y', selection='S')
    data, wyz, yz, z = matrices()
    target = {P(('W', 'Y', 'Z')): adapt_density(wyz, ('W', 'Y', 'Z')),
              P(('Y', 'Z')): adapt_density(yz, ('Y', 'Z')),
              P('Z'): adapt_density(z, ('Z',))}
    cfg = legacy['TCBConfig']()
    source = {P('Z', do='Y', domain='source'):
              adapt_callable(lambda z, y: legacy['_p_z_do_y'](z, y, cfg), ('Z', 'Y'), vectorized=False)}
    for value in (0, 1):
        expected = legacy['tcb_weights'](np.full_like(data['Y'], value), data['Y'], data['Z'],
                                         data['W'], wyz, yz, z, cfg).ravel()
        with pytest.warns(ApproximationWarning):
            result = compute_weights(analysis.weight_expression, data, {'Y': value},
                                     source_distributions=source, target_distributions=target,
                                     coupling='diagonal', normalize=False)
        assert_allclose(result.weights, expected, rtol=1e-12)


def test_historical_regression_via_automatic_more_do_and_rule2():
    legacy = load_numerics(EXPERIMENTS / 'syn_regression/nonIdent_tcb_syn_reg_30runs.py',
                           {'p_z_doy_v', '_as_scalar', 'tcb_weights'})
    graph = 'Y; Z; V; X; S; V -> Y; V -> Z; V -> X; Y -> Z; Z -> X; S -> X; Y <-> Z;'
    analysis = analyze(graph, outcomes='X', interventions='Y', selection='S', mode='more_do', simplify=True)
    rng = np.random.default_rng(37)
    data = {v: rng.normal(size=(25, 1)) for v in ('X', 'Y', 'Z', 'V')}
    def pv(points):
        return norm.pdf(np.asarray(points).reshape(-1))
    def pzv(points):
        matrix = np.column_stack(points) if isinstance(points, list) else np.asarray(points)
        return norm.pdf(matrix[:, 0], loc=matrix[:, 1], scale=2)*norm.pdf(matrix[:, 1])
    target = {P('V'): adapt_density(pv, ('V',)), P(('Z', 'V')): adapt_density(pzv, ('Z', 'V'))}
    source = {P('Z', given='V', do='Y', domain='source'):
              adapt_callable(legacy['p_z_doy_v'], ('Z', 'Y', 'V'))}
    for value in (-.5, 0., 1.):
        expected = legacy['tcb_weights'](value, data['V'], data['Z'], pv, pzv, legacy['p_z_doy_v']).ravel()
        result = compute_weights(analysis.weight_expression, data, {'Y': value},
                                 source_distributions=source, target_distributions=target)
        assert_allclose(result.weights, expected, rtol=1e-12)
