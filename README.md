# Transportable Causal Bootstrapping

`tcb` converts selection diagrams and revised sID output into transport formulas, inspectable transportable causal weight (TCW) expressions, numerical weights, and bootstrap samples. Its theoretical basis is the TCW derivation and TCB algorithm in `ch4.tex`. By default, it evaluates the full sum over pairs of observations.

## Installation and execution

The current supported range is Python 3.9–3.10, following the dependency constraints of `causalbootstrapping 0.2.5`. Run the following commands from this directory:

```powershell
python -m pip install .
python examples/quickstart.py
python -B -m pytest -q
```

You can also install the wheel in `dist/`. If all dependencies are already installed in your research environment, add `--no-deps` to the installation command. A recent pip version supporting PEP 660 also allows an editable installation with `python -m pip install -e .`. During local validation, unrelated pytest plugin autoloading was disabled, and `-B` prevented Python bytecode cache writes to the system environment:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -B -m pytest -q
```

## Basic workflow

For symbolic identification only, open
[the sID transport formula notebook](Demo/Demo00_sid_transport_formula.ipynb).
It demonstrates both identification modes without requiring data. Printing the returned formula produces only LaTeX, with marginalization written as `\sum`. In Jupyter, `display(formula)` renders the formula directly.

For an executed end-to-end example with plots, open
[the synthetic classification notebook](Demo/Demo01_nonident_tcb_synthetic_classification.ipynb).
See [Demo setup instructions](Demo/README.md) for optional notebook dependencies.

```python
import numpy as np
from tcb import (
    analyze, Probability, fit_target_distributions,
    compute_weights, bootstrap,
)

graph = '''Y; Z; W; X; S1; S2;
Y -> Z; Z -> W; W -> X;
Y <-> X; Y <-> Z; S1 -> X; S2 -> W;'''

analysis = analyze(
    graph, outcomes="X", interventions="Y",
    selection=("S1", "S2"), mode="standard",
)
print(analysis.describe())

# target_data: {"X": X, "Y": Y, "Z": Z, "W": W}
# Every array has N rows; X can have shape (N,d) or (N,H,W,C).
target = fit_target_distributions(
    analysis.weight_expression, target_data,
    method="histogram", bins={"W": 50, "Y": 0, "Z": 0},
)

# Example source-domain P(Z | do(Y)); replace it with your own model.
# The callback receives a dictionary of (B,d) arrays and returns one
# probability or density value per row.
def p_z_do_y(points):
    y, z = points["Y"][:, 0], points["Z"][:, 0]
    p1 = 0.2 + 0.6 * y
    return np.where(z == 1, p1, 1 - p1)

source = {Probability("Z", do="Y", domain="source"): p_z_do_y}
weights = compute_weights(
    analysis.weight_expression, target_data, {"Y": 1},
    source_distributions=source,
    target_distributions=target,
)
print(weights.effective_sample_size)
samples = bootstrap(target_data, weights, n_samples=1000, random_state=42)
X_train = samples.data["X"]
y_train = samples.data["intv_Y"]
```

`samples.indices` records the original data row indices. All arrays are sampled using the same indices. The original `Y` is retained, while intervention labels are stored in `intv_Y`; use the latter for training on the resampled data.

## High-level and low-level interfaces

| Layer | Interface | Output |
|---|---|---|
| Identification | `identify(graph, outcomes, interventions, selection, mode=...)` | `TransportFormula` |
| Symbolic weights | `derive_weights(formula)` | `WeightExpression` |
| Combined analysis | `analyze(...)` / `general_tcb_analysis(...)` | `Analysis`, including the formula and required distributions |
| Density estimation | `fit_target_distributions(expression, data, ...)` | Dictionary of target distributions |
| Weight computation | `compute_weights(expression, data, intervention, ...)` | `WeightResult` |
| Weight array only | `backend.weight_compute(...)` | `(N,)` array |
| Single-intervention resampling | `bootstrap(data, weights, ...)` | `BootstrapResult` |
| Multiple-intervention workflow | `transportable_bootstrap(analysis, data, interventions, ...)` | `TCBResult` |

`source_distributions` is a required argument for weight computation. Pass `{}` explicitly when no source distribution is needed. The library does not substitute target observational data for source interventional distributions.

All four fields of `Probability`—`variables`, `given`, `do`, and `domain`—participate in distribution matching. Consequently, `P_source(Z | do(Y))` and `P_source(Z | V,do(Y))` are distinct requirements. Inspect `analysis.required_source_distributions` for the exact distributions needed.

## Full empirical product and the historical approximation

* `coupling="product"` (default): following `ch4.tex`, when `W_R` is nonempty, compute `w_n = sum_j Psi(n,j)/(N*M)`. The same target dataset is used by default, so `M=N`. Bound symbols such as `Y'` and `Z'` retain their own scope, separate from free intervention symbols.
* `coupling="diagonal"`: compute the same-row expression `Psi(n,n)/N` to reproduce the older classification and bgMNIST experiments. This emits an `ApproximationWarning` and must not be interpreted as the full double-index formula.
* When `W_R` is empty: compute `Psi(n)/N` directly, without a second set of sample indices.

The `batch_size` and `integration_batch_size` parameters control memory usage. Duplicate integration rows, such as repeated values of binary `Z`, are grouped exactly using their frequencies, with no additional approximation. General continuous cases still require O(N²) pairs. You can explicitly provide `integration_data` to use another set of target-distribution samples, giving a scaling factor of `1/(N*M)`. You are responsible for ensuring that these samples represent the same target environment; subsampling introduces an additional numerical approximation.

Weights are normalized by default. `WeightResult.log_weights` always retains the unnormalized log weights, including the `1/N` or `1/(N*M)` factor. Use `normalize=False` or `.raw_weights` to retrieve the raw TCW. Resampling uses normalized probabilities, and the effective sample size (ESS) is available as a diagnostic.

## Reusing density estimators and source functions

Target density estimation directly reuses `fit_kde`, `fit_histogram`, and `fit_multinorm` from `causalbootstrapping.distEst_lib.MultivarContiDistributionEstimator`. Only the low-dimensional target joint distributions needed by the weights are fitted, and repeated terms share estimators.

* `adapt_density(pdf, ("W", "Y", "Z"))`: wrap an existing density function accepting a `(B,d)` matrix, with an explicit column order.
* `adapt_callable(p_z_doy_v, ("Z", "Y", "V"))`: wrap an existing vectorized source function with multiple positional arguments.
* `adapt_callable(fn, ("Z", "Y"), vectorized=False)`: wrap a function from an older script that processes one row at a time.

Histogram estimation retains the upstream **bin mass** semantics: for continuous variables, the returned values are not PDFs divided by bin width. Joint distributions sharing variables should use the same fitting data and consistent per-variable bins. A per-variable bin count of `0` follows the upstream category-count convention, which suits the examples' 0/1 variables; it is not a general PMF estimator for arbitrary strings or irregularly spaced categories. For continuous data, KDE or a user-defined density can be used.

Zero denominators, negative values, NaN, Inf, incorrect callback output shapes, and all-zero weights raise explicit errors. The library does not silently replace invalid values with a minimum value or clip weights. Under `product` coupling, cross-row combinations absent from the old scripts' evaluations may reveal insufficient support; check the density estimates and overlap assumptions.

## Discrete and continuous interventions

If the weights require `K(y-y_n)`, provide it explicitly through `kernels`:

```python
from tcb import delta_kernel, gaussian_kernel
kernels = {"Y": delta_kernel}          # Discrete intervention
kernels = {"Y": gaussian_kernel(0.2)}  # Continuous; choose the bandwidth
```

Each value in a single `intervention` dictionary is a fixed scalar or vector. Use the high-level workflow to process multiple interventions. In that workflow, `n_samples` is the total number of samples across all interventions, divided evenly by default; supply `sample_counts` to specify a count for each intervention.

```python
from tcb import observed_interventions, transportable_bootstrap
interventions, counts = observed_interventions(target_data, "Y")
result = transportable_bootstrap(
    analysis, target_data, interventions,
    source_distributions=source, target_distributions=target,
    sample_counts=counts, random_state=42,
)
```

For continuous `Y`, supply an explicit intervention grid, such as `[{'Y': y} for y in grid]`.

## Migrating the three experiments

`examples/experiment_adapters.py` provides functions that can replace the TCB stage of the existing experiments:

* `classification_tcb(...)`: classification, with intervention sample counts matching the original label counts.
* `bgmnist_tcb(...)`: the same computation, preserving image tensor dimensions.
* `regression_tcb(...)`: a specified continuous intervention grid, KDE estimates of `P*(V)` and `P*(Z,V)`, and a user-supplied source function.

These adapters do not read existing experiment results, train models, or modify the original experiment scripts. A complete runnable example is provided in `examples/quickstart.py`.

The regression adapter uses `mode="more_do", simplify=True`. It first obtains `P_source(Z | do(Y,V))` from sID, then applies the graphical m-separation check for do-calculus Rule 2 to exchange `do(V)` for conditioning on `V`. It also removes `Y` from the target outcome's conditioning set when the graph establishes the required conditional independence. This yields the paper's weight expression, `P_source(Z | V,do(Y))*P*(V)/(N*P*(Z,V))`. These transformations are recorded in `transport_formula.provenance`.

## Supported formulas and sID output checks

Automatic TCW derivation follows the conditions in Chapter 4: a unique target-domain outcome factor `P*(X | W_X)` must be extractable, and the remaining factors must not depend on the outcome. Formulas containing only a directly transported source outcome factor, multiple outcome dependencies, or nested ratios outside the supported representation raise `UnsupportedFormulaError`. This is distinct from `NotTransportableError`.

The library includes a snapshot of the supplied sID implementation and `LocalDOExpr`, with only the helper import changed to a relative package import. It retains `standard` / `more_do` modes and the fallback warning. Automatic identification preserves that implementation's behavior; it does not constitute a new soundness proof or validation for every graph. In particular:

* On the regression graph, the current standard sID output omits `V` from the source term. It must not silently be interpreted as the paper's conditional source distribution. Use the regression path described above.
* Some `more_do` outputs may have inconsistent marginal-variable and per-term do metadata. Numerical computation is rejected when unbound variables remain.
* Use `analyze(formula=TransportFormula(...))` to supply a verified formula explicitly. This is treated as user-provided mathematical input, not as a formula identified by sID.

See `docs/theory.md` for the mapping between equations and implementation, and `THIRD_PARTY.md` for provenance and reuse details.

## Release artifacts

The `dist/` directory contains the installable wheel, a standard source
distribution (`.tar.gz`), and a convenience `-source.zip` containing the demos
and supporting source files. `SHA256SUMS.txt` lists their checksums.
The wheel contains the Python library; use a source archive to access the Demo
notebooks. See [the release guide](docs/releasing.md) for build and validation
steps and the publication metadata still to be supplied.
