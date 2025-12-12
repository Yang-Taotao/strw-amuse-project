"""
Main script for running the 6-body encounter simulation and visualization.
"""

# from src.strw_amuse.sims import mc
# from src.strw_amuse.utils import logger

from src.strw_amuse.core import convert

if __name__ == ("__main__"):
    # logger.setup_logging()
    # result = mc.monte_carlo_19D(
    #     n_samples=100,
    # )
    file_path = f"./mc/mc_result_000{0}.npz"
    outcome_name = "Creative_ionized"
    n_bins = 100
    convert.visualize(file_path=file_path, outcome_name=outcome_name, n_bins=n_bins)
