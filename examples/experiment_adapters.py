"""Drop-in TCB stage replacements for the three supplied experiments.

These functions do not train models, read datasets or write experiment results.
They return TCBResult; train on result.data['X'] and result.data['intv_Y'].
"""
import numpy as np

from tcb import (analyze, adapt_callable, fit_target_distributions,
                 observed_interventions, transportable_bootstrap)


CLASSIFICATION_GRAPH = (
    'Y; Z; W; X; S1; S2; Y -> Z; Z -> W; W -> X; '
    'S1 -> X; S2 -> W; Y <-> X; Y <-> Z;'
)
REGRESSION_GRAPH = (
    'Y; Z; V; X; S; V -> Y; V -> Z; V -> X; Y -> Z; '
    'Z -> X; S -> X; Y <-> Z;'
)


def classification_tcb(data, p_z_do_y, *, bins_w=50, vectorized=False,
                       coupling='product', random_state=None):
    """For synthetic classification and bgMNIST; X may be an image tensor.

    Pass p_z_do_y(z, intervention_y), using a closure/partial for TCBConfig.
    coupling='diagonal' reproduces the old scripts' weighting approximation.
    """
    analysis = analyze(CLASSIFICATION_GRAPH, outcomes='X', interventions='Y', selection=('S1', 'S2'))
    target = fit_target_distributions(analysis.weight_expression, data, method='histogram',
                                      bins={'W': bins_w, 'Y': 0, 'Z': 0})
    source = {analysis.required_source_distributions[0]:
              adapt_callable(p_z_do_y, ('Z', 'Y'), vectorized=vectorized)}
    interventions, counts = observed_interventions(data, 'Y')
    return transportable_bootstrap(
        analysis, data, interventions, source_distributions=source,
        target_distributions=target, sample_counts=counts,
        coupling=coupling, random_state=random_state,
    )


bgmnist_tcb = classification_tcb


def regression_tcb(data, intervention_values, p_z_do_y_v, *, n_samples=None,
                   vectorized=True, random_state=None):
    """Automatic sID route matching ch4.tex's conditional-source regression.

    more_do gives P_source(Z | do(Y,V)); graph-checked rule 2 exchanges
    do(V) for V, while m-separation removes Y from P*(X | Y,V,Z).
    Source knowledge remains a required user-supplied argument.
    """
    analysis = analyze(REGRESSION_GRAPH, outcomes='X', interventions='Y', selection='S',
                       mode='more_do', simplify=True)
    target = fit_target_distributions(analysis.weight_expression, data, method='kde')
    source = {analysis.required_source_distributions[0]:
              adapt_callable(p_z_do_y_v, ('Z', 'Y', 'V'), vectorized=vectorized)}
    interventions = [{'Y': float(v)} for v in np.asarray(intervention_values).reshape(-1)]
    return transportable_bootstrap(
        analysis, data, interventions, source_distributions=source,
        target_distributions=target, n_samples=n_samples, random_state=random_state,
    )
