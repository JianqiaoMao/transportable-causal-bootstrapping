# Demonstrations

Choose a notebook:

- [Demo00_sid_transport_formula.ipynb](Demo00_sid_transport_formula.ipynb): symbolic sID identification only. Define a selection diagram, obtain a transport formula, compare the two modes, and handle an unsuccessful query. No dataset is required.
- [Demo01_nonident_tcb_synthetic_classification.ipynb](Demo01_nonident_tcb_synthetic_classification.ipynb): a complete synthetic classification experiment using the package.

From the repository root, install the optional notebook dependencies and launch Jupyter:

```sh
python -m pip install ".[demo]"
jupyter lab Demo
```

Use a Python 3.9–3.10 kernel containing the package dependencies, then select
**Restart Kernel and Run All Cells**. The notebook can be run from the package
root or its `Demo` directory; all paths are relative to that checkout.

## Synthetic classification experiment

The classification demonstration retains the original experiment's three environments,
4,000 training samples and 1,000 test samples per environment/regime,
seeds 111/222, and linear SVM with C=5. It produces nine trained classifiers
and 54 accuracy records, with formula, density requirements, weight diagnostics,
distribution shifts, decision boundaries, accuracy heatmaps and bar plots.

The default `COUPLING="product"` follows Chapter 4's complete empirical-product
weights. Set `COUPLING="diagonal"` and rerun to reproduce the older script's
same-row approximation. Resampling is now explicitly seeded; the original
script's model results are not expected to match bit for bit.

`synthetic_classification_data.py` contains the required SCM, checked against
the original generator for all three environments and both regimes. It avoids
imports from the external experiment directory and global random-state changes.
Latent variables are not used for fitting densities, computing weights or
training classifiers.

Outputs are embedded in the notebook. Additional CSV, JSON and PNG exports
are disabled by default; setting `SAVE_OUTPUTS=True` writes only beneath
`Demo/outputs/nonident_classification/`.
