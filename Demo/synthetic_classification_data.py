"""Portable version of the original not_ident_1 synthetic data generator.

Reference: source_scripts/experiment/syn_classification/syn_data_gen.py,
non_ident_discUYZ_contWX_syn_generator and the not_ident_1 wrapper branch.
The parameters, random draw order, and environment shifts are preserved.
A local RandomState replaces global np.random seeding. Latent U1 and U2
are returned for inspection only; neither is an input to TCB or the SVMs.
"""
import numpy as np
import pandas as pd


SOURCE_PARAMETERS = {"p_u1": 0.65, "alpha_z": 0.0, "beta_zy": 2.0, "beta_zu1": 0.75}
ENVIRONMENT_SHIFTS = {
    0: {"x1_shift": 0.0, "w_shift": 0.0, "x1_noise": 0.0, "w_noise": 0.0},
    1: {"x1_shift": 0.0, "w_shift": -1.5, "x1_noise": 0.0, "w_noise": 0.2},
    2: {"x1_shift": 5.0, "w_shift": 1.5, "x1_noise": 0.2, "w_noise": 0.2},
}


def generate_environment(env, confounded, n, seed):
    """Generate one environment/subset, matching the original random draws."""
    if env not in ENVIRONMENT_SHIFTS:
        raise ValueError("env must be 0, 1 or 2.")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(confounded, (bool, np.bool_)):
        raise ValueError("confounded must be boolean.")
    rng = np.random.RandomState(seed)
    shift = ENVIRONMENT_SHIFTS[env]
    u1 = (rng.rand(n) < SOURCE_PARAMETERS["p_u1"]).astype(int)
    u2 = (rng.rand(n) < 0.65).astype(int)
    beta_yu1, beta_yu2 = (0.5, 1.75) if confounded else (0.0, 0.0)
    logit_y = beta_yu1 * (2*u1 - 1) + beta_yu2 * (2*u2 - 1)
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit_y))).astype(int)
    logit_z = (SOURCE_PARAMETERS["alpha_z"] + SOURCE_PARAMETERS["beta_zy"] * (2*y - 1)
               + SOURCE_PARAMETERS["beta_zu1"] * (2*u1 - 1))
    z = (rng.rand(n) < 1 / (1 + np.exp(-logit_z))).astype(int)
    mu_w = np.where(z == 1, 4.0, -4.0).reshape(n, 1)
    mu_w[:, 0] += shift["w_shift"] + shift["w_noise"] * rng.randn(n)
    w = mu_w + rng.randn(n, 1)
    mu_x = np.zeros((n, 2))
    mu_x[:, 0] = np.where(w[:, 0] >= 0, 2.0, -2.0)
    mu_x[:, 0] += shift["x1_shift"] + shift["x1_noise"] * rng.randn(n)
    mu_x[:, 1] = np.where(u2 == 1, 2.0, -2.0)
    x = mu_x + rng.randn(n, 2)
    return pd.DataFrame({
        "X1": x[:, 0], "X2": x[:, 1], "Y": y, "Z": z, "W": w[:, 0],
        "U1": u1, "U2": u2, "env": env,
        "confounded": "confounded" if confounded else "non-confounded",
    })


def generate_dataset(n_per_subset, seed, envs=(0, 1, 2)):
    """Return six subsets by default: three environments x two regimes.

    As in the original experiment, both regimes within an environment use
    seed + env. Separate calls with seeds 111 and 222 give training and test
    datasets. n_per_subset is not the total size of the returned DataFrame.
    """
    return pd.concat([
        generate_environment(env, confounded, n_per_subset, seed + env)
        for env in envs for confounded in (False, True)
    ], ignore_index=True)
