"""
Configuation file for the STRW-AMUSE project.
- Contains global params for mc sampler and run args default.
- Contain param setup for an example run of the scripts.
- Contain file directory settings.
"""

from dataclasses import dataclass
from typing import Final  # <- Set global imutable cfg const
from pathlib import Path
import numpy as np

# configs - dir
# ================================================================================================ #
DIR_BASE: Final[Path] = Path(__file__).resolve().parents[3]
DIR_DATA: Final[Path] = DIR_BASE / "data"
DIR_MEDIA: Final[Path] = DIR_BASE / "media"
# ================================================================================================ #
OUTPUT_DIR_COLLISIONS: Final[Path] = DIR_DATA / "collisions"
OUTPUT_DIR_COLLISIONS_DIAGNOSTICS: Final[Path] = DIR_DATA / "collisions_diagnostics"
OUTPUT_DIR_COLLISIONS_OUTCOMES: Final[Path] = DIR_DATA / "collisions_outcomes"
OUTPUT_DIR_FINAL_STATES: Final[Path] = DIR_DATA / "final_states"
OUTPUT_DIR_GIF: Final[Path] = DIR_MEDIA / "gif"
OUTPUT_DIR_IMG: Final[Path] = DIR_MEDIA / "img"
OUTPUT_DIR_LOGS: Final[Path] = DIR_DATA / "logs"
OUTPUT_DIR_MC: Final[Path] = DIR_DATA / "mc"
OUTPUT_DIR_OUTCOMES: Final[Path] = DIR_DATA / "outcomes"
OUTPUT_DIR_SAMPLER: Final[Path] = DIR_MEDIA / 'img' / 'param'
OUTPUT_DIR_SNAPSHOTS: Final[Path] = DIR_MEDIA / "snapshots"
# ================================================================================================ #

# configs - global param
# ================================================================================================ #
N_DIMS: Final = 19
N_SAMPLES: Final = 10000
SEED: Final = 42

BOUNDS: Final = np.array(
    [
        [0.00, 0.99],  # ecc 1
        [0.00, 0.99],  # ecc 2
        [0.00, 0.99],  # ecc 3
        [1.50, 7.00],  # sep 1
        [1.50, 7.00],  # sep 2
        [1.50, 7.00],  # sep 3
        [0.10, 5.00],  # v_mag 1
        [0.10, 5.00],  # v_mag 2
        [0.00, 10.00],  # impact_parameter 1
        [0.00, 10.00],  # impact_parameter 2
        [0.00, np.pi / 2],  # theta 1
        [0.00, np.pi / 2],  # theta 2
        [0.00, 2 * np.pi],  # phi 1
        [0.00, 2 * np.pi],  # phi 2
        [0.00, 2 * np.pi],  # psi 1
        [0.00, 2 * np.pi],  # psi 2
        [0.00, 2 * np.pi],  # true_anomalies 1
        [0.00, 2 * np.pi],  # true_anomalies 2
        [0.00, 2 * np.pi],  # true_anomalies 3
    ]
)

# configs - Plotter param selection
# ================================================================================================ #
PARAM_CROSS_SECTION: Final = {
    r"impact parameter $\rho ~\left(\mathrm{AU}\right)$": [
        "impact_parameter_0",
        "impact_parameter_1",
    ],
    r"eccentricity $e$": ["ecc_0", "ecc_1", "ecc_2"],
    r"separation $\mathrm{sep} ~\left(\mathrm{AU}\right)$": ["sep_0", "sep_1", "sep_2"],
    r"$\mathrm{v}_\mathrm{mag} \sim \mathrm{v}_\mathrm{escape} ~\left(\mathrm{km}~\mathrm{s}^{-1}\right)$": [
        "v_mag_0",
        "v_mag_1",
    ],
}
PARAM_CORNER: Final = [
    "ecc_0",
    "ecc_1",
    "ecc_2",
    "sep_0",
    "sep_1",
    "sep_2",
    "impact_parameter_0",
    "impact_parameter_1",
    "v_mag_0",
    "v_mag_1",
]
# ================================================================================================ #


# configs - MC dataclass
# ================================================================================================ #
@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results data class."""

    samples: np.ndarray
    param_names: list
    distances: np.ndarray
    weights: np.ndarray
    all_star_outcomes: np.ndarray
    all_star_weights: np.ndarray
    sample_ids: np.ndarray
    weighted_counts: np.ndarray
    probabilities: np.ndarray
    unique_outcomes: np.ndarray


# ================================================================================================ #


# configs - example run params
# ================================================================================================ #
# set ref param for no collison case
PARAM_REF: Final = (
    [30, 20, 10],  # sep_ref | AU
    [0.0, 0.0, 0.0],  # ecc_ref
    [2, 1],  # v_mag_ref
    [0.0, 10.0],  # impact_parameter_ref | AU
    [np.pi / 3, np.pi / 4],  # theta_ref
    [np.pi, np.pi / 2],  # phi_ref
    [np.pi / 2, np.pi],  # psi_ref
    [100, 100],  # distance_ref | AU
    [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4],  # true_anomalies_ref
    "ref_case",  # label_ref
)
# set test param for collison case with ionized collision
PARAM_TEST: Final = (
    [30, 20, 10],  # sep_ref | AU
    [0.0, 0.0, 0.0],  # ecc_ref
    [0.2, 0.2],  # v_mag_ref
    [0.0, 0.0],  # impact_parameter_ref | AU
    [np.pi / 4, np.pi / 4],  # theta_ref
    [np.pi / 2, np.pi / 2],  # phi_ref
    [np.pi / 2, np.pi / 2],  # psi_ref
    [50, 50],  # distance_ref | AU
    [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4],  # true_anomalies_ref
    "test_case",  # label_ref
)
# ================================================================================================ #
