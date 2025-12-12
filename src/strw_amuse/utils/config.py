"""
Configuation file for the STRW-AMUSE project.
- Contain param setup for an example run of the scripts.
- Contain file directory settings.
"""

from dataclasses import dataclass

import numpy as np

# configs - dir
# ================================================================================================ #
OUTPUT_DIR_GIF = "./media/gif"
OUTPUT_DIR_IMG = "./media/img"
OUTPUT_DIR_COLLISIONS = "./data/collisions"
OUTPUT_DIR_COLLISIONS_DIAGNOSTICS = "./data/collisions_diagnostics"
OUTPUT_DIR_COLLISIONS_OUTCOMES = "./data/collisions_outcomes"
OUTPUT_DIR_OUTCOMES = "./data/outcomes"
OUTPUT_DIR_FINAL_STATES = "./data/final_states"
OUTPUT_DIR_SNAPSHOTS = "./media/snapshots"
OUTPUT_DIR_LOGS = "./data/logs"
OUTPUT_DIR_MC = "./data/mc"
OUTPUT_DIR_SAMPLER = './media/img/param'
# ================================================================================================ #

# configs - global param
# ================================================================================================ #
N_DIMS = 19
N_SAMPLES = 10000
SEED = 42
BOUNDS = np.array(
    [
        [0.00, 0.99],  # ecc 1
        [0.00, 0.99],  # ecc 2
        [0.00, 0.99],  # ecc 3
        [2.00, 7.00],  # sep 1
        [2.00, 7.00],  # sep 2
        [2.00, 7.00],  # sep 3
        [0.10, 1.00],  # v_mag 1
        [0.10, 1.00],  # v_mag 2
        [0.00, 5.00],  # impact_parameter 1
        [0.00, 5.00],  # impact_parameter 2
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
PARAM_CROSS_SECTION = {
    "impact parameter": ["impact_parameter_0", "impact_parameter_1"],
    "eccentricity": ["ecc_0", "ecc_1", "ecc_2"],
    "separation": ["sep_0", "sep_1", "sep_2"],
    "v_mag": ["v_mag_0", "v_mag_1"],
}
PARAM_CORNER = [
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
PARAM_REF = (
    # sep_ref | AU
    [30, 20, 10],
    # ecc_ref
    [0.0, 0.0, 0.0],
    # v_mag_ref
    [2, 1],
    # impact_parameter_ref | AU
    [0.0, 10.0],
    # theta_ref
    [np.pi / 3, np.pi / 4],
    # phi_ref
    [np.pi, np.pi / 2],
    # psi_ref
    [np.pi / 2, np.pi],
    # distance_ref | AU
    [100, 100],
    # true_anomalies_ref
    [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4],
    # label_ref
    "ref_case",
)
# set test param for collison case with ionized collision
PARAM_TEST = (
    # sep_ref | AU
    [30, 20, 10],
    # ecc_ref
    [0.0, 0.0, 0.0],
    # v_mag_ref
    [0.2, 0.2],
    # impact_parameter_ref | AU
    [0.0, 0.0],
    # theta_ref
    [np.pi / 4, np.pi / 4],
    # phi_ref
    [np.pi / 2, np.pi / 2],
    # psi_ref
    [np.pi / 2, np.pi / 2],
    # distance_ref | AU
    [50, 50],
    # true_anomalies_ref
    [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4],
    # label_ref
    "test_case",
)
# ================================================================================================ #
