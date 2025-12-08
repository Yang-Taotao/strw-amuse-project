"""
Main script for running the 6-body encounter simulation and visualization.
"""

from src.strw_amuse.config import OUTPUT_DIR_LOGS, PARAM_REF, PARAM_TEST
from src.strw_amuse.logging_config import setup_logging

setup_logging(log_dir=OUTPUT_DIR_LOGS)

from src.strw_amuse.plotter import plot_gif, plot_trajectory
from src.strw_amuse.run_simulation import run_6_body_simulation

frames_ref, outcome_ref = run_6_body_simulation(*PARAM_REF)

plot_gif(frames=frames_ref, run_label="ref_case")
plot_trajectory(frames=frames_ref, run_label="ref_case")

frames_test, outcome_test = run_6_body_simulation(*PARAM_TEST)

plot_gif(frames=frames_test, run_label="test_case")
plot_trajectory(frames=frames_test, run_label="test_case")
