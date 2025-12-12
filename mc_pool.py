"""
Main script for running the 6-body encounter simulation and visualization.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import os
import multiprocessing

from src.strw_amuse.sims import mc, mc_alt
from src.strw_amuse.utils import logger

if __name__ == ("__main__"):
    logger.setup_logging()
    multiprocessing.set_start_method("spawn", force=True)
    # result = mc.monte_carlo_19D(
    #     n_samples=100,
    # )
    n_cores = 30
    # MC config for local test
    n_samples = 100  # number of 19D samples for THIS run
    n_jobs = 1  # single job locally
    job_idx = 0  # single segment

    result = mc_alt.monte_carlo_19D(
        n_samples=n_samples,
        n_jobs=n_jobs,
        job_idx=job_idx,
        verbose=True,
        n_workers=n_cores,
        save=True,
    )
    # Quick summary
    print("\n=== Local MC test finished ===")
    print(f"Samples used: {result.samples.shape[0]}")
    print(f"Unique outcomes: {len(result.unique_outcomes)}")
    if result.probabilities.size:
        print("Outcome probabilities:")
        for u, p in zip(result.unique_outcomes, result.probabilities):
            print(f"{u}: {p:.3f}")
    else:
        print("No collisions or no successful outcomes recorded.")
