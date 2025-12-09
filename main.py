"""
Main script for running the 6-body encounter simulation and visualization.
"""

from src.strw_amuse.utils import logger, checker

if __name__ in ("__main__"):
    logger.setup_logging()
    checker.check_sim_example()
