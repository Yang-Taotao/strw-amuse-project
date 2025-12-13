"""
Main script for running the 6-body encounter simulation and visualization.
"""

import multiprocessing

from src.strw_amuse.plots import visualization
from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger

if __name__ == "__main__":
    # init logger
    logger.setup_logging()

    # force mp
    multiprocessing.set_start_method("spawn", force=True)

    # get mc results
    result = mc.monte_carlo_19D(n_samples=10, n_workers=10)

    # local repo
    file_path = f"./data/mc/mc_result_000{0}.npz"
    outcome_name = "Creative_ionized"
    n_bins = 10

    # visualiza mc results
    visualization.visualize(file_path=file_path, outcome_name=outcome_name, n_bins=n_bins)
