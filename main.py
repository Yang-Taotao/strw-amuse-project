"""
Main script for running the 6-body encounter simulation and visualization.
"""

from src.strw_amuse.sims import mc
from src.strw_amuse.utils import logger

if __name__ == ("__main__"):
    logger.setup_logging()
    result = mc.monte_carlo_19D(
        n_samples=100,
    )
