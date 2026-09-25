"""Transportable causal bootstrapping, grounded in thesis Chapter 4."""
from .symbolic import Probability, TransportFormula, WeightExpression, derive_weights
from .identification import identify, from_sid, simplify_formula
from .density import fit_target_distributions, adapt_density, adapt_callable
from .backend import (compute_weights, weight_compute, bootstrap, WeightResult,
                      BootstrapResult, delta_kernel, gaussian_kernel)
from .workflows import (analyze, general_tcb_analysis, Analysis, observed_interventions,
                        transportable_bootstrap, TCBResult)
from .exceptions import (TCBError, NotTransportableError, UnsupportedFormulaError,
                         MissingDistributionError, PositivityError, ApproximationWarning)

__version__ = "0.1.0"

__all__ = [
    "Probability", "TransportFormula", "WeightExpression", "derive_weights",
    "identify", "from_sid", "simplify_formula", "fit_target_distributions", "adapt_density", "adapt_callable",
    "compute_weights", "weight_compute", "bootstrap", "WeightResult", "BootstrapResult",
    "delta_kernel", "gaussian_kernel", "analyze", "general_tcb_analysis", "Analysis",
    "observed_interventions", "transportable_bootstrap", "TCBResult", "TCBError",
    "NotTransportableError", "UnsupportedFormulaError", "MissingDistributionError",
    "PositivityError", "ApproximationWarning",
]
