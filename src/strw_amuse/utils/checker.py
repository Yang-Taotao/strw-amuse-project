"""
Example run checker for `run_6body_simulation()`.
"""

from ..plots.plotter import plot_gif, plot_trajectory
from ..sims.run_sim import run_6_body_simulation
from .config import PARAM_REF, PARAM_TEST


def check_sim_example(param_ref: tuple = PARAM_REF, param_test: tuple = PARAM_TEST) -> None:
    """Simple `run_6body_simulation()` run check and visualization"""
    frames_ref, _ = run_6_body_simulation(*param_ref)
    frames_test, _ = run_6_body_simulation(*param_test)

    plot_gif(frames=frames_ref, run_label="ref_case")
    plot_trajectory(frames=frames_ref, run_label="ref_case")

    plot_gif(frames=frames_test, run_label="test_case")
    plot_trajectory(frames=frames_test, run_label="test_case")
    return None
