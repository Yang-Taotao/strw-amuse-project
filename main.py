"""
Main script for running the 6-body encounter simulation and visualization.
"""

from src.strw_amuse.utils import logger  # , checker
from src.strw_amuse.sims import mc  # , monte_carlo

if __name__ in ("__main__"):
    logger.setup_logging()
    # checker.check_sim_example()
    # monte_carlo.monte_carlo_19D(n_samples=1, n_cores=1)
    mc.monte_carlo_19D(n_samples=1000)
