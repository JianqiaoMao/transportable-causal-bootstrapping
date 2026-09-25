import copy
import itertools
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tcb import (
    Probability as P, TransportFormula, analyze, identify, compute_weights,
    fit_target_distributions, adapt_density, adapt_callable, bootstrap,
    observed_interventions, transportable_bootstrap, delta_kernel, gaussian_kernel,
    simplify_formula, NotTransportableError, UnsupportedFormulaError, MissingDistributionError,
    PositivityError, ApproximationWarning,
)


CLASSIFICATION_GRAPH = (
    'Y; Z; W; X; S1; S2; Y -> Z; Z -> W; W -> X; '
    'S1 -> X; S2 -> W; Y <-> X; Y <-> Z;'
)


@pytest.fixture
def classification():
    analysis = analyze(CLASSIFICATION_GRAPH, outcomes='X', interventions='Y', selection=('S1', 'S2'))
    # Full binary support with deliberately dependent Y,Z,W and unequal counts.
    triples = np.array(list(itertools.product([0, 1], repeat=3)))
    counts = np.array([4, 1, 2, 7, 1, 5, 8, 2])
    rows = np.repeat(triples, counts, axis=0)
    data = {v: rows[:, i:i+1] for i, v in enumerate(('W', 'Y', 'Z'))}
    data['X'] = np.arange(len(rows) * 12).reshape(len(rows), 3, 4, 1)
    target = fit_target_distributions(analysis.weight_expression, data,
                                     method='histogram', bins={'W': 0, 'Y': 0, 'Z': 0})
    source = {P(('Z',), do=('Y',), domain='source'):
              lambda p: np.where(p['Z'][:, 0] == 1, 0.2 + 0.6*p['Y'][:, 0], 0.8 - 0.6*p['Y'][:, 0])}
    return analysis, data, target, source


def _paper_pairwise(data, target, y):
    n = len(data['Y'])
    wyz, yz, z = target[P(('W', 'Y', 'Z'))], target[P(('Y', 'Z'))], target[P(('Z',))]
    expected = np.zeros(n)
    diagonal = np.zeros(n)
    for i in range(n):
        observed = {v: data[v][i:i+1] for v in ('W', 'Y', 'Z')}
        observed_conditional = np.asarray(wyz(observed) / yz(observed)).item()
        for j in range(n):
            mixed = {'W': data['W'][i:i+1], 'Y': np.array([[y]]), 'Z': data['Z'][j:j+1]}
            intervention_conditional = np.asarray(wyz(mixed) / yz(mixed)).item()
            causal = 0.2 + 0.6*y if mixed['Z'][0, 0] == 1 else 0.8 - 0.6*y
            pair = intervention_conditional * causal / (observed_conditional * np.asarray(z(mixed)).item())
            expected[i] += pair / n**2
            if i == j:
                diagonal[i] = pair / n
    return expected, diagonal


def test_sid_to_chapter4_expression(classification):
    a, _, _, _ = classification
    assert a.weight_expression.w_x == ('W', "Y'", "Z'")
    assert a.weight_expression.w_r == ('Z',)
    assert a.weight_expression.kernel_variables == ()
    assert a.required_source_distributions == (P('Z', do='Y', domain='source'),)
    assert all('X' not in p.symbols for p in a.required_target_distributions)
    assert 'sum_{j=1..M}' in str(a.weight_expression)
    assert a.transport_formula.aliases == {"Y'": 'Y', "Z'": 'Z'}


@pytest.mark.parametrize('y', [0, 1])
def test_full_product_equals_independent_chapter4_double_loop(classification, y):
    a, d, target, source = classification
    expected, diagonal = _paper_pairwise(d, target, y)
    result = compute_weights(a.weight_expression, d, {'Y': y}, source_distributions=source,
                             target_distributions=target, normalize=False,
                             batch_size=7, integration_batch_size=3)
    assert_allclose(result.weights, expected, rtol=1e-12)
    assert not np.allclose(expected/expected.sum(), diagonal/diagonal.sum())
    assert 1 <= result.effective_sample_size <= len(d['Y'])


def test_duplicate_compression_and_chunks_preserve_exact_sum(classification):
    a, d, t, s = classification
    results = [compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s,
               target_distributions=t, compress_integration=compress,
               batch_size=b, integration_batch_size=j) for compress, b, j in
               [(True, 1, 1), (False, 5, 7), (False, 100, 100)]]
    for result in results[1:]:
        assert_allclose(result.log_weights, results[0].log_weights, rtol=1e-12)


def test_separate_integration_data_uses_correct_nm_scale(classification):
    a, d, t, s = classification
    original = compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s,
                                target_distributions=t, normalize=False)
    doubled = compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s,
                               target_distributions=t, normalize=False,
                               integration_data={'Z': np.repeat(d['Z'], 2, axis=0)})
    assert doubled.integration_size == 2 * original.integration_size
    assert_allclose(original.weights, doubled.weights)


def test_log_space_normalization_avoids_ratio_overflow(classification):
    a, d, t, s = classification
    t = {**t, P('Z'): lambda p: 1e-300}
    s = {next(iter(s)): lambda p: 1e300}
    result = compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s,
                             target_distributions=t)
    assert np.isfinite(result.weights).all()
    assert_allclose(result.weights.sum(), 1)
    assert np.all(result.log_weights > 1000)


def test_diagonal_reproduces_historical_classification_weights(classification):
    a, d, t, s = classification
    _, expected = _paper_pairwise(d, t, 1)
    with pytest.warns(ApproximationWarning, match='same-row'):
        result = compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s,
                                 target_distributions=t, coupling='diagonal', normalize=False)
    assert_allclose(result.weights, expected)


def regression_formula():
    return TransportFormula(
        outcomes='X', interventions='Y', marginal=('Z', 'V'),
        numerator=(P('X', given=('Z', 'V')), P('V'), P('Z', given='V', do='Y', domain='source')),
        provenance='ch4.tex: transport formula for synthetic regression',
    )


def test_conditional_source_and_regression_equation():
    a = analyze(formula=regression_formula())
    assert a.weight_expression.w_r == ()
    assert a.weight_expression.kernel_variables == ()
    rng = np.random.default_rng(34)
    d = {v: rng.normal(size=(60, 1)) for v in ('X', 'Y', 'Z', 'V')}
    target = fit_target_distributions(a.weight_expression, d, method='kde')
    from scipy.stats import norm
    fn = lambda z, y, v: norm.pdf(z, loc=-1.5*y+v, scale=np.sqrt(2))
    key = P('Z', given='V', do='Y', domain='source')
    source = {key: adapt_callable(fn, ('Z', 'Y', 'V'))}
    result = compute_weights(a.weight_expression, d, {'Y': 0.7}, source_distributions=source,
                             target_distributions=target, normalize=False)
    expected = fn(d['Z'], 0.7, d['V']).ravel()*target[P('V')](d)/target[P(('Z', 'V'))](d)/60
    assert_allclose(result.weights, expected)
    with pytest.raises(MissingDistributionError, match='P_source'):
        compute_weights(a.weight_expression, d, {'Y': 0}, target_distributions=target,
                        source_distributions={P('Z', do='Y', domain='source'): source[key]})


def test_automatic_regression_more_do_matches_verified_paper_formula():
    graph = 'Y; Z; V; X; S; V -> Y; V -> Z; V -> X; Y -> Z; Z -> X; S -> X; Y <-> Z;'
    automatic = analyze(graph, outcomes='X', interventions='Y', selection='S', mode='more_do', simplify=True)
    manual = analyze(formula=regression_formula())
    assert automatic.weight_expression.numerator == manual.weight_expression.numerator
    assert automatic.weight_expression.denominator == manual.weight_expression.denominator
    assert automatic.required_source_distributions == (P('Z', given='V', do='Y', domain='source'),)
    assert automatic.weight_expression.kernel_variables == ()
    assert 'rule 2' in automatic.transport_formula.provenance


def test_rule2_retains_bidirected_edges_in_separation_check():
    formula = TransportFormula('X', 'Y',
        (P('X', given=('V', 'Z')), P('V'), P('Z', do=('Y', 'V'), domain='source')),
        marginal=('V', 'Z'))
    graph = 'Y; Z; V; X; V -> Z; Y -> Z; Z -> X; V <-> Z;'
    simplified = simplify_formula(formula, graph)
    source = [p for p in simplified.numerator if p.domain == 'source'][0]
    assert source.do == ('V', 'Y')
    assert source.given == ()


def test_multiple_interventions_and_fixed_vector_values():
    formula = TransportFormula('X', ('A', 'B'),
        (P('X', given='Z'), P('Z', do=('A', 'B'), domain='source')), marginal='Z')
    a = analyze(formula=formula)
    d = {'X': np.arange(8).reshape(4, 2), 'Z': np.array([0, 1, 0, 1]),
         'A': np.zeros((4, 2)), 'B': np.zeros(4)}
    key = P('Z', do=('A', 'B'), domain='source')
    def source(points):
        assert points['A'].shape[1] == 2
        assert_allclose(points['A'], np.tile([1, 2], (len(points['A']), 1)))
        return np.where(points['Z'][:, 0] == 1, .75, .25)
    result = compute_weights(a.weight_expression, d, {'A': [1, 2], 'B': 3},
        source_distributions={key: source}, target_distributions={P('Z'): lambda p: .5})
    assert_allclose(result.weights, [.125, .375, .125, .375])
    sampled = bootstrap(d, result, n_samples=7, random_state=4)
    assert sampled.data['intv_A'].shape == (7, 2)
    assert_allclose(sampled.data['intv_B'], 3)


def test_kernel_backdoor_and_multichar_names():
    a = analyze('Treatment; Outcome; Confounder; Confounder -> Treatment; '
                'Confounder -> Outcome; Treatment -> Outcome;',
                outcomes='Outcome', interventions='Treatment')
    assert a.weight_expression.kernel_variables == ('Treatment',)
    d = {'Outcome': np.arange(4), 'Treatment': np.array([0, 1, 0, 1]),
         'Confounder': np.array([0, 0, 1, 1])}
    t = fit_target_distributions(a.weight_expression, d, method='histogram',
                                 bins={'Treatment': 0, 'Confounder': 0})
    with pytest.raises(ValueError, match='kernels'):
        compute_weights(a.weight_expression, d, {'Treatment': 1}, source_distributions={}, target_distributions=t)
    result = compute_weights(a.weight_expression, d, {'Treatment': 1},
                             source_distributions={}, target_distributions=t,
                             kernels={'Treatment': delta_kernel})
    assert_allclose(result.weights, [0, 0.5, 0, 0.5])


def test_gaussian_kernel():
    values = gaussian_kernel(0.5)(np.array([[0.], [0.5]]), np.zeros((2, 1)))
    assert_allclose(values, np.array([1, np.exp(-0.5)])/(0.5*np.sqrt(2*np.pi)))
    with pytest.raises(ValueError):
        gaussian_kernel(0)


def test_source_and_target_registries_never_conflate(classification):
    a, d, t, _ = classification
    with pytest.raises(MissingDistributionError, match='source'):
        compute_weights(a.weight_expression, d, {'Y': 0}, source_distributions={}, target_distributions=t)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf])
def test_invalid_denominator_rejected(classification, bad):
    a, d, t, s = classification
    t = dict(t)
    t[P('Z')] = lambda points: bad
    with pytest.raises(PositivityError):
        compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s, target_distributions=t)


def test_callback_shape_is_not_silently_flattened(classification):
    a, d, t, s = classification
    s = {next(iter(s)): lambda points: np.ones((len(points['Z']), 2))}
    with pytest.raises(ValueError, match='must return'):
        compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s, target_distributions=t)


def test_zero_source_weights_rejected(classification):
    a, d, t, s = classification
    with pytest.raises(PositivityError, match='All TCW'):
        compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions={next(iter(s)): lambda p: 0},
                        target_distributions=t)


def test_bootstrap_images_alignment_labels_reproducibility(classification):
    a, d, t, s = classification
    d = {**d, "keep'": np.arange(len(d['Y']))}
    result = compute_weights(a.weight_expression, d, {'Y': 1}, source_distributions=s, target_distributions=t)
    before = copy.deepcopy(d)
    global_rng_state = np.random.get_state()
    first = bootstrap(d, result, n_samples=120, random_state=22)
    second = bootstrap(d, result, n_samples=120, random_state=22)
    assert_array_equal(first.indices, second.indices)
    after_rng_state = np.random.get_state()
    assert_array_equal(global_rng_state[1], after_rng_state[1])
    assert global_rng_state[2:] == after_rng_state[2:]
    assert first.data['X'].shape == (120, 3, 4, 1)
    assert_array_equal(first.data['intv_Y'], np.ones((120, 1)))
    for key in d:
        assert_array_equal(first.data[key], d[key][first.indices])
        assert_array_equal(d[key], before[key])
    with pytest.raises(ValueError, match='must match'):
        bootstrap(d, result, intervention={'Y': 0})


def test_high_level_multi_intervention_counts(classification):
    a, d, t, s = classification
    interventions, counts = observed_interventions(d, 'Y')
    result = transportable_bootstrap(a, d, interventions, source_distributions=s,
                                     target_distributions=t, sample_counts=counts, random_state=31)
    assert len(result.indices) == len(d['Y'])
    assert [int(np.sum(result.data['intv_Y'] == i)) for i in (0, 1)] == counts
    assert_array_equal(result.data['X'], d['X'][result.indices])
    assert_allclose([r.weights.sum() for r in result.weight_results], [1, 1])


def test_not_transportable_and_unsupported_are_distinct():
    with pytest.raises(NotTransportableError):
        identify('X; Y; S; Y -> X; Y <-> X; S -> X;', 'X', 'Y', 'S')
    with pytest.raises(UnsupportedFormulaError, match='target observational factor'):
        analyze(formula=TransportFormula('X', 'Y', (P('X', do='Y', domain='source'),)))
    with pytest.raises(ValueError, match='mode'):
        identify('X; Y; Y -> X;', 'X', 'Y', mode='invalid')


def test_sid_modes_and_unbound_do_metadata():
    # Both identification modes are exposed independently of TCW applicability.
    standard = identify('X; Y; S; Y -> X; S -> Y;', 'X', 'Y', 'S')
    more = identify('X; Y; S; Y -> X; S -> Y;', 'X', 'Y', 'S', mode='more_do')
    assert all(p.domain == 'target' for p in standard.numerator)
    assert any(p.domain == 'source' for p in more.numerator)
    with pytest.raises(UnsupportedFormulaError, match='Unbound'):
        TransportFormula('X', 'Y', (P('X', do=('Y', 'Z'), domain='source'),))


def test_scalar_legacy_callback_adapter():
    source = adapt_callable(lambda z, y: np.array([0.8 if z[0] == y[0] else 0.2]),
                            ('Z', 'Y'), vectorized=False)
    assert_allclose(source({'Z': np.array([[0], [1]]), 'Y': np.array([[1], [1]])}), [0.2, 0.8])


def test_no_intervention_returns_uniform_empirical_weights():
    a = analyze('X; Z; Z -> X;', outcomes='X', interventions=())
    d = {'X': np.arange(5), 'Z': np.arange(5)}
    r = compute_weights(a.weight_expression, d, {}, source_distributions={}, target_distributions={})
    assert_allclose(r.weights, np.full(5, 0.2))
