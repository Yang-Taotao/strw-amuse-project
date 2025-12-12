"""
Visualization utilities for plotting MC run saved results.
"""

from ..core import convert
from ..plots import plotter
from ..utils.config import PARAM_CORNER, PARAM_CROSS_SECTION


def visualize(
    file_path: str,
    outcome_name: str,
    param_groups: dict = PARAM_CROSS_SECTION,
    param_subset: list = PARAM_CORNER,
    n_bins: int = 10,
) -> None:
    """
    Wrapper function for visualizing MC run results.
    - Call `plotter.plot_cross_section()`
    - Call `plotter.corner_for_outcome()`

    Args:
        file_path (str): File path to MC results `.npz` file.
        outcome_name (str): Name of the outcome to plot against.
        param_groups (dict, optional): Params data to load for `plot_cross_section()`.
            Defaults to `PARAM_CROSS_SECTION`.
        param_subset (list, optional): List of param data to load for `corner_for_outcome()`.
            Defaults to `PARAM_CORNER`.
        n_bins (int, optional): Number of bins to use for plotting. Defaults to 10.

    Returns:
        None: No returns expected from this wrapper function.
    """
    data_result = convert.to_MonteCarloResults(file_path=file_path)
    plotter.plot_cross_section(
        mc_result=data_result,
        outcome_name=outcome_name,
        param_groups=param_groups,
        n_bins=n_bins,
    )
    plotter.corner_for_outcome(
        mc_result=data_result,
        outcome_name=outcome_name,
        param_subset=param_subset,
        bins=n_bins,
    )
    return None
