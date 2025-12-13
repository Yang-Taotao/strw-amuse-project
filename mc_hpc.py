"""
Docstring for mc_hpc. Used as main script call for hpc slurm jobs.
"""

import logging
import multiprocessing
import os

from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger

if __name__ == "__main__":
    # logger init
    logger.setup_logging()
    logs = logging.getLogger(__name__)

    # force mp
    multiprocessing.set_start_method("spawn", force=True)

    # local repo
    total_samples = 100000
    n_jobs = 10  # must match SBATCH --array
    samples_per_job = total_samples // n_jobs

    # set job by slurm array index (0-based)
    job_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    # samples to run per job
    n_samples = samples_per_job
    # set node specific n_workers ~ n_cores
    n_workers = 20

    # get mc results and save
    result = mc.monte_carlo_19D(
        n_samples=n_samples,
        n_jobs=n_jobs,
        job_idx=job_idx,
        verbose=True,
        n_workers=n_workers,
        save=True,
    )

    # do a quick summary <- not necessary for hpc runs
    logs.info("\n=== Local MC test finished ===")
    logs.info(f"Samples used: {result.samples.shape[0]}")
    logs.info(f"Unique outcomes: {len(result.unique_outcomes)}")
    if result.probabilities.size:
        logs.info("Outcome probabilities:")
        for u, p in zip(result.unique_outcomes, result.probabilities):
            logs.info(f"{u}: {p:.3f}")
    else:
        logs.info("No collisions or no successful outcomes recorded.")
