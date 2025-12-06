"""
Configuation file for the STRW-AMUSE project.
- Contain param setup for an example run of the scripts.
- Contain file directory settings.
"""

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
OUTPUT_DIR_LOGS = "./data/logs"
OUTPUT_DIR_SNAPSHOTS = "./media/snapshots"
# ================================================================================================ #

# configs - example run params
# ================================================================================================ #
# set ref param for no collison case
PARAM_REF = (
    # sep_ref | AU
    [30, 20, 10],
    # ecc_ref
    [0.0, 0.0, 0],
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
    [0.0, 0.0, 0],
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
