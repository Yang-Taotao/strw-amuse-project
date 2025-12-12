"""
MC sampler test script.
"""

import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube

from src.strw_amuse.utils.config import (
    OUTPUT_DIR_SAMPLER,
    N_DIMS,
    N_SAMPLES,
    BOUNDS,
    SEED,
)


def gen_nd_samples(
    n_samples: int = N_SAMPLES,
    n_dims: int = N_DIMS,
    bounds: np.ndarray = BOUNDS,
    save_dir: str = OUTPUT_DIR_SAMPLER,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Param sampler supporting `n` dimensions with two methods.
    - Uniform: `numpy.random.uniform()`
    - LHS: `scipy.stats.qmc.LatinHyperCube()`

    Args:
        n_samples (int, optional): Defaults to N_SAMPLES.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.
        save_dir (str, optional): Defaults to OUTPUT_DIR_SAMPLER.

    Returns:
        tuple[np.ndarray, np.ndarray]: Samples (Uniform), Samples (LHS)
    """
    # init
    os.makedirs(save_dir, exist_ok=True)

    # local repo
    low, high = nd_bounds(bounds)
    rng = np.random.default_rng(seed)

    # numpy sampler
    samples_np = rng.uniform(low, high, (n_samples, n_dims))

    # scipy sampler <- choose this
    sampler_sp = LatinHypercube(d=n_dims, rng=rng)
    samples_sp = sampler_sp.random(n=n_samples) * (high - low) + low

    return samples_np, samples_sp


def nd_bounds(bounds: np.ndarray = BOUNDS) -> tuple[np.ndarray, np.ndarray]:
    """
    Get `low` and `high` param bounds.

    Args:
        bounds (np.ndarray, optional): Defaults to BOUNDS.

    Returns:
        tuple[np.ndarray, np.ndarray]: bounds (`low`), bounds (`high`)
    """
    low, high = bounds[:, 0], bounds[:, 1]
    return low, high


def nd_coverage(samples: np.ndarray, n_dims: int = N_DIMS, bounds: np.ndarray = BOUNDS) -> list:
    """
    Get param sapce `coverage` for some `samples`

    Args:
        samples (np.ndarray): Generated samples.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.

    Returns:
        list: Coverage of some `samples`
    """
    low, high = nd_bounds(bounds)
    coverage = [np.ptp(samples[:, i]) / (high[i] - low[i]) for i in range(n_dims)]
    return coverage
