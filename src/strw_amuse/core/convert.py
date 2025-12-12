import numpy as np
from src.strw_amuse.sims import mc_alt
from src.strw_amuse.utils.config import PARAM_CROSS_SECTION, PARAM_CORNER
from src.strw_amuse.plots import plotter


def to_MonteCarloResults(file_path) -> mc_alt.MonteCarloResult:

    data = np.load(file_path, allow_pickle=True)

    result = mc_alt.MonteCarloResult(
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


def visualize(
    file_path, outcome_name, param_groups=PARAM_CROSS_SECTION, param_subset=PARAM_CORNER, n_bins=10
) -> None:
    data_result = to_MonteCarloResults(file_path=file_path)
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
