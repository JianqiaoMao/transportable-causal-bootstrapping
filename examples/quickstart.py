"""Small runnable examples; no data downloads and no model training."""
import numpy as np
from scipy.stats import norm

from experiment_adapters import classification_tcb, regression_tcb


def main():
    rng = np.random.default_rng(7)
    n = 160
    y = rng.integers(0, 2, (n, 1))
    z = rng.integers(0, 2, (n, 1))
    w = rng.integers(0, 2, (n, 1))
    data = {'Y': y, 'Z': z, 'W': w, 'X': rng.normal(size=(n, 8, 8, 1))}

    def source_z(z, y):
        return np.where(z == 1, 0.2 + 0.6*y, 0.8 - 0.6*y)

    classification = classification_tcb(data, source_z, bins_w=0, vectorized=True, random_state=11)
    print(classification.analysis.describe())
    print('Image output:', classification.data['X'].shape,
          'intervention labels:', classification.data['intv_Y'].shape)
    print('ESS:', [round(r.effective_sample_size, 2) for r in classification.weight_results])

    regression_data = {v: rng.normal(size=(80, 1)) for v in ('X', 'Y', 'Z', 'V')}

    def source_z_given_v(z, y, v):
        return norm.pdf(z, loc=-1.5*y + v, scale=np.sqrt(2))

    regression = regression_tcb(regression_data, [-0.5, 0, 0.5], source_z_given_v,
                                n_samples=81, random_state=12)
    print(regression.analysis.describe())
    print('Regression output:', regression.data['X'].shape)


if __name__ == '__main__':
    main()
