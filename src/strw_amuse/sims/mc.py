"""
Monte Carlo utilities for cross section calculation.
Uses 19D LHS samples from src.strw_amuse.core.sampler and runs 6-body sims.
"""

import logging
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from ..core import sampler
from ..utils.config import OUTPUT_DIR_MC, SEED, MonteCarloResult
from .run_sim import run_6_body_simulation

logger = logging.getLogger(__name__)


def _run_single_simulation(args):
    """
    Helper to run one 6-body sim from a single 19D sample row.
    """
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


def _monte_carlo_core(
    samples: np.ndarray, verbose: bool = True, n_workers: int | None = None
) -> MonteCarloResult:
    """
    Core MC logic operating on a given samples array of shape (n_samples, 19).
    Handles parallel execution and aggregation, but not sampling or saving.
    """
    n_samples = samples.shape[0]

    # Reconstruct param_names as in your original mc.py
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
    param_names = [
        f"{label}_{i}" for label, count in zip(param_labels, param_counts) for i in range(count)
    ]

    distances = np.full((n_samples, 2), 100.0)
    weights = np.ones(n_samples)

    pool_args = list(zip(samples, distances, weights))

    # Parallel execution: processes
    if n_workers is None:
        n_workers = min(max(cpu_count() - 1, 1), 16)

    logger.info(f"MC: starting {n_samples} sims on {n_workers} workers")

    if n_workers == 1:
        results = []
        for args in tqdm(pool_args, total=n_samples, disable=not verbose):
            outcome, w = _run_single_simulation(args)
            results.append((outcome, w))
    else:
        with Pool(processes=n_workers) as pool:
            results_iter = pool.imap_unordered(_run_single_simulation, pool_args, chunksize=1)
            results = list(tqdm(results_iter, total=n_samples, disable=not verbose))

    # Flatten stars with >=1 collision and store sample mapping
    flat_data = []
    flat_weights = []
    flat_ids = []

    for sample_idx, (outcome, w) in enumerate(results):
        if outcome == "simulation_failed":
            continue
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
            flat_ids.append(sample_idx)

    dtype = [
        ("star_key", "uint64"),
        ("collisions", "int32"),
        ("n_companions", "int32"),
        ("mass_Msun", "float64"),
        ("outcome", "U32"),
    ]

    if flat_data:
        all_star_outcomes = np.array(flat_data, dtype=dtype)
        all_star_weights = np.array(flat_weights)
        sample_ids = np.array(flat_ids)

        unique_outcomes, inv_idx = np.unique(all_star_outcomes["outcome"], return_inverse=True)
        weighted_counts = np.bincount(inv_idx, weights=all_star_weights)
        probabilities = weighted_counts / weighted_counts.sum()
    else:
        logger.warning("MC: no successful outcomes with collisions.")
        all_star_outcomes = np.array([], dtype=dtype)
        all_star_weights = np.array([])
        sample_ids = np.array([])
        unique_outcomes = np.array([], dtype="U32")
        weighted_counts = np.array([])
        probabilities = np.array([])

    return MonteCarloResult(
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


def monte_carlo_19D(
    n_samples: int,
    n_jobs: int = 1,
    job_idx: int = 0,
    verbose: bool = True,
    n_workers: int | None = None,
    save: bool = True,
    seed: int = SEED,
) -> MonteCarloResult:
    """
    Monte Carlo 19D simulation for one job segment using sampler.gen_nd_samples.

    - If n_jobs == 1: uses all n_samples.
    - If n_jobs > 1: assumes global n_samples_per_job * n_jobs design and takes
      the slice for this job_idx.

    Intended usage with Slurm arrays:
        #SBATCH --array=0-9
        job_i = int(os.environ['SLURM_ARRAY_TASK_ID'])
        monte_carlo_19D(n_samples=100, n_jobs=10, job_idx=job_i, ...)
    """
    # Generate full design, then slice for this job
    # Here we interpret n_samples as per-job samples.
    total_samples = n_samples * n_jobs
    _, all_samples = sampler.gen_nd_samples(n_samples=total_samples, seed=seed)
    logger.info("MC: sample gen at shape %s.", all_samples.shape)

    start = job_idx * n_samples
    stop = start + n_samples
    samples = all_samples[start:stop]

    logger.info(f"MC: job_idx={job_idx} using samples[{start}:{stop}] out of {total_samples}")

    result = _monte_carlo_core(samples, verbose=verbose, n_workers=n_workers)

    if save:
        os.makedirs(OUTPUT_DIR_MC, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR_MC, f"mc_result_{job_idx:04d}.npz")
        np.savez(
            out_path,
            samples=result.samples,
            param_names=np.array(result.param_names, dtype="U32"),
            distances=result.distances,
            weights=result.weights,
            all_star_outcomes=result.all_star_outcomes,
            all_star_weights=result.all_star_weights,
            sample_ids=result.sample_ids,
            weighted_counts=result.weighted_counts,
            probabilities=result.probabilities,
            unique_outcomes=result.unique_outcomes,
        )
        logger.info(f"MC: saved result to {out_path}")

    return result
