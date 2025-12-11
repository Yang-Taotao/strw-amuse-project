"""
Main script for running the 6-body encounter simulation and visualization.
"""

import os
from src.strw_amuse.sims import mc  # , monte_carlo
from src.strw_amuse.utils import logger  # , checker

if __name__ == ("__main__"):
    logger.setup_logging()
    # checker.check_sim_example()
    # monte_carlo.monte_carlo_19D(n_samples=1, n_cores=1)
    job_i = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    result = mc.monte_carlo_19D(
        n_samples=100,
        n_jobs=100,
        job_idx=job_i,
        save=True,
    )
