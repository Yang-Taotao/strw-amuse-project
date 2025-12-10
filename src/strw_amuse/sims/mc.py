"""
Vectorized Monte Carlo utilities for cross section calculation.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.strw_amuse.utils import config

from .run_simulation import run_6_body_simulation

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
#  Vectorized Latin Hypercube
# ----------------------------------------------------------------------
def sample_19D_lhs(n_samples, rng=None, n_jobs=1, job_idx=0):
    """
    Generate stratified Latin Hypercube samples in the fixed 19D parameter space,
    optionally splitting the parameter ranges by job index.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate for this job.
    rng : np.random.Generator
        Random number generator.
    n_jobs : int
        Total number of jobs splitting the parameter ranges.
    job_idx : int
        Index of this job (0-based).

    Returns
    -------
    samples : np.ndarray, shape (n_samples, 19)
    param_names : list of str
    distances : np.ndarray, shape (n_samples, 2)
    weights : np.ndarray, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    param_counts = np.array([3, 3, 2, 2, 2, 2, 2, 3])
    param_labels = [
        "ecc",
        "sep",
        "v_mag",
        "impact_parameter",
        "theta",
        "phi",
        "psi",
        "true_anomalies",
    ]
    lower_bounds = np.array([0.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    upper_bounds = np.array([0.99, 7.0, 1.0, 5.0, np.pi / 2, 2 * np.pi, 2 * np.pi, 2 * np.pi])

    # Expand bounds vectorized
    param_names = [
        f"{label}_{i}" for label, count in zip(param_labels, param_counts) for i in range(count)
    ]
    param_lows = np.repeat(lower_bounds, param_counts)
    param_highs = np.repeat(upper_bounds, param_counts)
    n_params = param_lows.size

    # Split ranges according to job index
    ranges = param_highs - param_lows
    segment_size = ranges / n_jobs
    job_lows = param_lows + job_idx * segment_size
    job_highs = job_lows + segment_size

    # Latin Hypercube sampling inside this job's segment
    lhs = (rng.uniform(size=(n_samples, n_params)) + np.arange(n_samples)[:, None]) / n_samples
    samples = job_lows + lhs * (job_highs - job_lows)

    distances = np.full((n_samples, 2), 100.0)
    weights = np.ones(n_samples)

    return samples, param_names, distances, weights


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
def _run_single_simulation(args):
    sample_row, distance_row, w = args

    ecc = sample_row[0:3]
    sep = sample_row[3:6]
    v_mag = sample_row[6:8]
    impact = sample_row[8:10]
    theta = sample_row[10:12]
    phi = sample_row[12:14]
    psi = sample_row[14:16]
    anomalies = sample_row[16:19]

    try:
        _, outcome = run_6_body_simulation(
            sep=sep,
            ecc=ecc,
            v_mag=v_mag,
            impact_parameter=impact,
            theta=theta,
            phi=phi,
            psi=psi,
            distance=distance_row,
            true_anomalies=anomalies,
            run_label="MC",
        )
        return outcome, w
    except Exception:
        return "simulation_failed", w


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    samples: np.ndarray
    param_names: list
    distances: np.ndarray
    weights: np.ndarray
    all_star_outcomes: np.ndarray
    all_star_weights: np.ndarray
    sample_ids: np.ndarray
    weighted_counts: np.ndarray
    probabilities: np.ndarray
    unique_outcomes: np.ndarray


# ----------------------------------------------------------------------
# Main Monte Carlo (vectorized post-processing)
# ----------------------------------------------------------------------
def monte_carlo_19D(n_samples, n_jobs=1, job_idx=0, verbose=True, save=True):
    """
    Monte Carlo 19D simulation for a single job segment.

    Parameters
    ----------
    n_samples : int
        Number of samples for this job.
    verbose : bool
        Show progress bar.
    save: bool
        Set to True to save results
    n_jobs : int
        Total number of jobs dividing the parameter space.
    job_idx : int
        Index of this job (0-based).

    Returns
    -------
    MonteCarloResult
    """
    # Generate samples for this job's segment
    samples, param_names, distances, weights = sample_19D_lhs(
        n_samples, n_jobs=n_jobs, job_idx=job_idx
    )

    pool_args = list(zip(samples, distances, weights))
    results = []

    # Run sequentially
    for args in tqdm(pool_args, total=n_samples, disable=not verbose):
        outcome, w = _run_single_simulation(args)
        results.append((outcome, w))

    # ----------------------------------------
    # Vectorized extraction of star outcomes
    # ----------------------------------------
    valid_mask = np.array([r[0] != "simulation_failed" for r in results])
    valid_indices = np.nonzero(valid_mask)[0]

    flat_data = []
    flat_weights = []
    flat_ids = []

    for idx in valid_indices:
        outcome, w = results[idx]
        stars = [s for s in outcome if s["collisions"] >= 1]

        for s in stars:
            flat_data.append(
                (
                    s["star_key"],
                    s["collisions"],
                    s["n_companions"],
                    s["mass_Msun"],
                    s["outcome"],
                )
            )
            flat_weights.append(w)
            flat_ids.append(idx)

    dtype = [
        ("star_key", "uint64"),
        ("collisions", "int32"),
        ("n_companions", "int32"),
        ("mass_Msun", "float64"),
        ("outcome", "U32"),
    ]

    all_star_outcomes = np.array(flat_data, dtype=dtype)
    all_star_weights = np.array(flat_weights)
    sample_ids = np.array(flat_ids)

    # ----------------------------------------
    # Vectorized weighted outcome counts
    # ----------------------------------------
    unique_outcomes, inv = np.unique(all_star_outcomes["outcome"], return_inverse=True)
    weighted_counts = np.bincount(inv, weights=all_star_weights)
    probabilities = weighted_counts / weighted_counts.sum()

    result = MonteCarloResult(
        samples=samples,
        param_names=param_names,
        distances=distances,
        weights=weights,
        all_star_outcomes=all_star_outcomes,
        all_star_weights=all_star_weights,
        sample_ids=sample_ids,
        weighted_counts=weighted_counts,
        probabilities=probabilities,
        unique_outcomes=unique_outcomes,
    )

    if save is True:
        result_dict = {
            'samples': result.samples,  # (nsamples, 19) param array
            'distances': result.distances,  # (nsamples, 2)
            'weights': result.weights,  # (nsamples,)
            'allstaroutcomes': result.all_star_outcomes,  # Structured array
            'allstarweights': result.all_star_weights,
            'sampleids': result.sample_ids,
            'probabilities': result.probabilities,  # Summary stats
            'uniqueoutcomes': result.unique_outcomes,
            'paramnames': np.array(result.param_names),  # For plotting labels
            'metadata': {'nsamples': n_samples, 'njobs': n_jobs, 'jobidx': job_idx},
        }

        file_name = os.path.join(config.OUTPUT_DIR_MC, f"MC_{job_idx:04d}.npy")
        np.save(file_name, np.array(result_dict, dtype=object))
        logger.info("MC: job [%s] results saved.", job_idx)
        logger.info("###===### END OF RUN ###===### END OF RUN ###===### END OF RUN ###===###")

    return result
