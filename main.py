"""
Main script for running the 6-body encounter simulation and visualization.
"""

from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger  # , checker

if __name__ == ("__main__"):
    logger.setup_logging()
    # checker.check_sim_example()
    result = mc.monte_carlo_19D(
        n_samples=100,
    )
