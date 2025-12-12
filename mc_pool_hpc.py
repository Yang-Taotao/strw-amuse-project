import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import os
import logging
from src.strw_amuse.sims import mc_alt
from src.strw_amuse.utils import logger
import multiprocessing

if __name__ == "__main__":
    logger.setup_logging()
    logs = logging.getLogger(__name__)
    multiprocessing.set_start_method("spawn", force=True)

    total_samples = 100000
    n_jobs = 10  # must match SBATCH --array
    samples_per_job = total_samples // n_jobs

    # Slurm array index (0-based)
    job_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    n_samples = samples_per_job  # samples this job will actually run
    n_workers = 20  # or 30, depending on node

    result = mc_alt.monte_carlo_19D(
        n_samples=n_samples,
        n_jobs=n_jobs,
        job_idx=job_idx,
        verbose=True,
        n_workers=n_workers,
        save=True,
    )
    # Quick summary
    logs.info("\n=== Local MC test finished ===")
    logs.info(f"Samples used: {result.samples.shape[0]}")
    logs.info(f"Unique outcomes: {len(result.unique_outcomes)}")
    if result.probabilities.size:
        logs.info("Outcome probabilities:")
        for u, p in zip(result.unique_outcomes, result.probabilities):
            logs.info(f"{u}: {p:.3f}")
    else:
        logs.info("No collisions or no successful outcomes recorded.")
