"""
Main script for running the 6-body encounter simulation and visualization.
"""

import src.strw_amuse as project

if __name__ in ("__main__"):
    project.utils.logger.setup_logging()
    project.utils.checker.check_sim_example()
