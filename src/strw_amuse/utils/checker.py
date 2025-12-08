"""
Example run checker for `run_6body_simulation()`.
"""

from .config import PARAM_REF, PARAM_TEST
from ..plotter import plot_gif, plot_trajectory
from ..run_simulation import run_6_body_simulation


def check_sim_example(param_a: tuple = PARAM_REF, param_b: tuple = PARAM_TEST):
    """Simple `run_6body_simulation()` run check and visualization"""
    frames_ref, outcome_ref = run_6_body_simulation(*PARAM_REF)
    frames_test, outcome_test = run_6_body_simulation(*PARAM_TEST)

    plot_gif(frames=frames_ref, run_label="ref_case")
    plot_trajectory(frames=frames_ref, run_label="ref_case")

    plot_gif(frames=frames_test, run_label="test_case")
    plot_trajectory(frames=frames_test, run_label="test_case")
