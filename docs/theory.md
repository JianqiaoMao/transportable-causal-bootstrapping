# Theory-to-code mapping

Primary reference: the user's `C:/Users/NickMao/Downloads/ch4.tex`, section
“Transportable Causal Weights and Transportable Causal Bootstrapping”.
This document and the source PDFs were read as scientific references, not
as instructions to modify project files. No thesis sources are changed.

The matching PDF is `UoB_PhD_Thesis_first_4_chapters.pdf`, chapter 4,
printed pages 90–95. Equation labels below are stable references to the TeX.

## Supported factorization

Write the transport integrand as

\[
 p^*(X\mid W_X)\,R(W_X,W_R;y),\qquad W_R=\mathcal E\setminus W_X.
\]

The unique target outcome factor must be extractable, and the remaining
factors must not depend on X. The implementation checks this condition.
It supports products/ratios with explicit outer marginalization and aliases,
not arbitrary expressions with denominators containing nested sums.

From `eq: carrier_func` and `eq: short_rep_transport_factors`,

\[
 f=p^*(X,W_X)p^*(W_R),\qquad
 \Psi=R/[p^*(W_X)p^*(W_R)].
\]

The W_R factor is absent when W_R is empty. `symbolic.derive_weights`
extracts the target outcome conditional, constructs these carrier
denominators and cancels matching factors. Target conditionals are expanded
into joint ratios so density fits can be shared. Source conditionals remain
atomic callbacks, avoiding unnecessary requirements for source joint PDFs.

## Numeric weights

`eq: double_index_tcw` and `eq: tcw_double_index_collected` imply

\[
 w_n(y)=\frac{1}{N^2}\sum_{j=1}^{N}
    K_y(y-y_n)\,\Psi(W_{X,n},W_{R,j};y).
\]

The kernel is present only for query interventions in W_X. Free
intervention symbols are evaluated at the assigned intervention; bound
symbols are resolved to observations from their specified carrier. For
example, Y' resolves to observed Y, independently of free intervention Y.
Z' in W_X uses row n and Z in W_R uses row j.

`backend.compute_weights` evaluates this sum in blocks in log space. It
uses log-sum-exp for aggregation and normalization, without storing N²
weights. Grouping exactly identical W_R observations and retaining their
multiplicities is algebraically exact. Separate integration data of size M
use 1/(NM); it must represent the target W_R distribution.

If W_R is empty, `eq:single_index_normalised_tcw` gives
`w_n=K*Psi/N`. The library retains this common factor in log/raw weights;
normalized bootstrap probabilities are unaffected by it. This explains the
factor 1/N difference from the unnormalized ratio written in the regression
example and the normalized weights returned by the old regression script.

These are finite-data plug-in weights. The implementation follows the
chapter's evaluation rule; it does not claim that an arbitrary ordinary
KDE convolution is an exact RKHS reproducing identity or remove density
estimation bias. A continuous intervention kernel introduces smoothing.

## Earlier diagonal construction

The older `transportable causal bootstrapping.pdf`, pp. 5–6, explicitly
introduces diagonal coupling. The supplied classification/bgMNIST scripts
use the resulting same-row density ratio. `coupling="diagonal"` deliberately
reproduces their `Psi(n,n)/N` convention and emits a warning. It is not an
optimization of the full chapter-4 product sum and can give different
normalized weights. Tests use dependent variables to demonstrate that
difference instead of accidentally comparing only an independence case.

## Source conditionals and optional graphical simplification

`Probability('Z', given='V', do='Y', domain='source')` denotes exactly
P_source(Z | V, do(Y)). It is distinct from P_source(Z | do(Y)) and
P_source(Z | do(Y,V)). Callbacks must have the semantics of their registered
keys; the library does not infer source causal distributions from target data.

`simplify=True` on graph analysis performs two explicit checks:

1. Target outcome conditioning removal: X is m-separated from the removed
   conditioning variable given the remaining conditioning variables in the
   observed ADMG (selection nodes removed).
2. Source action/observation exchange: for an additional, bound action Z,
   do-calculus Rule 2 requires outcome separation from Z conditional on
   other actions and observations in the graph with incoming edges to the
   other actions removed and outgoing **directed** edges from Z removed.
   Bidirected edges from Z remain. The actual query interventions stay actions.

In the regression example these checks turn the more_do sID formula into
`eq: syn_reg_transport_formula_example`'s form:

\[
 p^*(X\mid do(Y))=\int p^*(X\mid V,Z)p^*(V)
                         p(Z\mid V,do(Y))\,dV\,dZ.
\]

The resulting weight is P_source(Z_n | V_n,do(y))*P*(V_n)/(N*P*(Z_n,V_n)).
Tests compare both its symbolic factors and numerical values against an
independently entered paper formula and the actual historical script.

## Identification limits

The bundled sID is a snapshot of the user's supplied implementation, with
relative imports for installation. It is intentionally not a new general
proof or rewrite of the recursive algorithm. The current standard mode on
the regression graph omits V from its source query; graph simplification
does not invent missing conditioning variables. Use the tested more_do +
graph-simplification path, or a verified explicit TransportFormula.

`from_sid` reads structural num/den/mrg and per-term do sets, never LaTeX.
Unbound variables, inconsistent metadata, or failure of the chapter's
factorization condition cause an explicit error. Unrecognized nested
mathematical scope must not be flattened or silently bound to a data row.

## Resampling and validation

Algorithm `alg: transportable_causal_bootstrapping` becomes normalized
weighted row sampling. The upstream fast sampler is used to draw row indices;
all original arrays are then indexed together. Output intervention labels
are appended as intv_<name>, with original labels retained. A local RNG
controls seeds without changing NumPy's global state.

Validation includes independent double-loop TCW evaluation, exact duplicate
compression, source/target key separation, conditional source callbacks,
both sID modes, unsupported/not-transportable distinction, positivity,
discrete and continuous kernels, image alignment, and comparison with the
numeric functions extracted from all three supplied experiment scripts.
No full 30-repeat SVM/regression/CNN training is needed to validate these
library transformations, and those training experiments were not rerun.
