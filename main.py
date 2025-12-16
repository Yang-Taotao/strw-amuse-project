"""
Main script for running the 6-body encounter simulation and visualization.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import multiprocessing

from src.strw_amuse.core import convert
from src.strw_amuse.plots import visualization
from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger, config


if __name__ == "__main__":
    # make sure to activate the conda env for this run
    # init logger
    logger.setup_logging()

    # force mp
    multiprocessing.set_start_method("spawn", force=True)

    # get mc results for smaller samples monte carlo simulation ~ 1-2 min real time
    result = mc.monte_carlo_19D(n_samples=10, n_workers=10)

    # local dir assignment for results checcking
    dir_path = config.DIR_BASE / "mc"
    file_path = config.DIR_BASE / "mc" / "combined_mc.npz"
    outcome_name = "Creative_ionized"
    n_bins = 500

    # combine all mc results `.npz` files into singular file
    convert.to_one_npz(dir_path=dir_path, file_path=file_path)

    # visualiza mc results
    visualization.visualize(file_path=file_path, outcome_name=outcome_name, n_bins=n_bins)
