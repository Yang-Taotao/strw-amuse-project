"""
Conversion utilities for converting `.npz` files into `MonteCarloResult` dataclass.
"""

import numpy as np

from src.strw_amuse.utils.config import MonteCarloResult


def to_MonteCarloResults(file_path: str) -> MonteCarloResult:
    """
    MC results converter
    - Load mc results from `file_path`
    - Convert to `MonteCarloResult` dataclass

    Args:
        file_path (str): File path of `.npz` file.

    Returns:
        MonteCarloResult: MonteCarloResult dataclass for analysis.
    """

    data = np.load(file_path, allow_pickle=True)

    result = MonteCarloResult(
        samples=data["samples"],
        param_names=list(data["param_names"]),
        distances=data["distances"],
        weights=data["weights"],
        all_star_outcomes=data["all_star_outcomes"],
        all_star_weights=data["all_star_weights"],
        sample_ids=data["sample_ids"],
        weighted_counts=data["weighted_counts"],
        probabilities=data["probabilities"],
        unique_outcomes=data["unique_outcomes"],
    )
    return result
