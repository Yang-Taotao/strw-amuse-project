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

from src.strw_amuse.core import convert, sampler
from src.strw_amuse.plots import plotter, visualization
from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger, config


if __name__ == "__main__":
    # init logger
    logger.setup_logging()

    # force mp
    multiprocessing.set_start_method("spawn", force=True)

    # get mc results
    result = mc.monte_carlo_19D(n_samples=10, n_workers=10)

    # local repo
    dir_path = config.DIR_BASE / "mc" / "mc"
    file_path = config.DIR_BASE / "mc" / "mc" / "combined_mc.npz"
    outcome_name = "Creative_ionized"
    n_bins = 500

    # combine all mc results into singular file
    convert.to_one_npz(dir_path=dir_path, file_path=file_path)

    # visualiza mc results
    visualization.visualize(file_path=file_path, outcome_name=outcome_name, n_bins=n_bins)
    samples_np, samples_sp = sampler.gen_nd_samples()
    plotter.sampler_nd_coverage_plot(samples_np, samples_sp)
