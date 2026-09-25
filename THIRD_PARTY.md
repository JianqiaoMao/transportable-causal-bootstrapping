# Provenance and reuse

* `src/tcb/_sid/algorithm.py` is a snapshot of the user's
  `code/sid_algorithm/sID_expr.py`, including both modes and its warning/
  fallback. The expression helper uses a relative package import. Separation
  queries call the package's own helper rather than the locally added
  grapl ADMG.ismsep method, which is absent from the published dependency. `src/tcb/_sid/expressions.py` is the supplied `expr_extend.py`.
  Updating the original files does not automatically update this snapshot.
* Density estimation imports `MultivarContiDistributionEstimator` from the
  installed `causalbootstrapping.distEst_lib`; it does not copy that library.
* Sampling imports `causalbootstrapping.backend.cw_bootstrapper`, using its
  fast sampler on an index array to preserve all original column names and
  tensor shapes. Its robust sampler uses a different RNG path and is not
  exposed by this version.
* The layered workflow/backend layout and importance-ratio calculation are
  informed by the author's [CausalBootstrapping repository](https://github.com/JianqiaoMao/CausalBootstrapping).
  The upstream weight builder is not called because it does not preserve
  the source/target distinction and per-factor intervention scope needed
  here. Weight derivation and the empirical-product evaluator are implemented
  separately from chapter 4.
* Compatibility was tested with installed causalbootstrapping 0.2.5,
  NumPy 1.23.5, SciPy 1.13.1 and Python 3.9.12. The upstream declared NumPy
  requirement is newer than that existing environment; a fresh dependency
  resolution will follow the declared requirements.

* `src/tcb/_graph.py` implements m-separation through a canonical latent DAG
  and ancestral moral graph, without modifying the installed grapl package.
  See the [Ananke canonical DAG documentation](https://ananke.readthedocs.io/en/stable/ananke.graphs.html)
  for the latent expansion and the [NetworkX separation documentation](https://networkx.org/documentation/stable/reference/algorithms/d_separation.html)
  for separation terminology. This helper is independently implemented.

* Release validation also uses a clean Python 3.9 environment resolved from the
  declared dependencies: causalbootstrapping 0.2.6, NumPy 1.26.4, SciPy 1.13.1,
  and the unmodified published grapl-causal 1.6.1.
