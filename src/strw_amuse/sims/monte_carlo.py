"""
Monte Carlo simulation utilities for cross section calculation.
"""

import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from .run_simulation import run_6_body_simulation

logger = logging.getLogger(__name__)


def sample_19D_lhs(n_samples, rng=None):
    """
    Generate stratified Latin Hypercube samples in the fixed 19D parameter space
    for 3 binaries + 2 incoming binaries encounters.
    Returns:
        - samples: np.ndarray of shape (n_samples, 19)
        - param_names: list of 19 strings corresponding to each column
        - distances: np.ndarray of shape (n_samples, 2), fixed outer distances
        - weights: np.ndarray of shape (n_samples,), all ones
    """

    if rng is None:
        rng = np.random.default_rng()
        logger.info("MC: lhs: `rng` not found, default `rng` assigned.")

    # --- Define independent parameter counts ---
    # ecc, sep, v_mag, impact, theta, phi, psi, anomalies
    param_counts = np.array([3, 3, 2, 2, 2, 2, 2, 3])
    param_labels = [
        "ecc",
        "sep",
        "v_mag",
        "impact_parameter",
        "theta",
        "phi",
        "psi",
        "true_anomalies",
    ]

    # --- Define bounds as arrays ---
    lower_bounds = np.array([0.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    upper_bounds = np.array([0.99, 7.0, 1.0, 5.0, np.pi / 2, 2 * np.pi, 2 * np.pi, 2 * np.pi])

    # Flatten counts and bounds
    n_params = int(np.sum(param_counts))
    param_names = []
    param_lows = np.zeros(n_params)
    param_highs = np.zeros(n_params)
    idx = 0
    for label, count, low, high in zip(param_labels, param_counts, lower_bounds, upper_bounds):
        for i in range(count):
            param_names.append(f"{label}_{i}")
            param_lows[idx] = low
            param_highs[idx] = high
            idx += 1

    # --- Latin Hypercube sampling ---
    lhs = rng.uniform(size=(n_samples, n_params))
    for j in range(n_params):
        lhs[:, j] = (lhs[:, j] + np.arange(n_samples)) / n_samples

    # Scale to parameter ranges
    samples = param_lows + lhs * (param_highs - param_lows)
    logger.info("MC: lhs: `samples`sampled with shape %s.", samples.shape)

    # Fixed distances and weights
    distances = np.full((n_samples, 2), 100.0)
    logger.info("MC: `distances` sampled with shape %s.", distances.shape)
    weights = np.ones(n_samples)
    logger.info("MC: lhs: `weights` sampled with shape %s.", distances.shape)

    return samples, param_names, distances, weights


def _run_single_simulation(args):
    """
    Runs a single simulation given array slices.
    args = (sample_row, distances_row, weight)
    """
    sample_row, distances_row, w = args
    logger.info("MC: run_single: local var unpacked to `args`.")

    # Column indices based on param_names in sample_19D_lhs
    # param_names = ["ecc_0","ecc_1","ecc_2", "sep_0","sep_1","sep_2",
    #                "v_mag_0","v_mag_1", "impact_parameter_0","impact_parameter_1",
    #                "theta_0","theta_1", "phi_0","phi_1", "psi_0","psi_1",
    #                "true_anomalies_0","true_anomalies_1","true_anomalies_2"]

    ecc = sample_row[0:3]
    logger.info("MC: run_single: built `ecc` at shape %s.", ecc.shape)
    sep = sample_row[3:6]
    logger.info("MC: run_single: built `sep` at shape %s.", sep.shape)
    v_mag = sample_row[6:8]
    logger.info("MC: run_single: built `v_mag` at shape %s.", v_mag.shape)
    impact_parameter = sample_row[8:10]
    logger.info("MC: run_single: built `impact_parameter` at shape %s.", impact_parameter.shape)
    theta = sample_row[10:12]
    logger.info("MC: run_single: built `theta` at shape %s.", theta.shape)
    phi = sample_row[12:14]
    logger.info("MC: run_single: built `phi` at shape %s.", phi.shape)
    psi = sample_row[14:16]
    logger.info("MC: run_single: built `psi` at shape %s.", psi.shape)
    true_anomalies = sample_row[16:19]
    logger.info("MC: run_single: built `true_anomalies` at shape %s.", true_anomalies.shape)
    distance = distances_row
    logger.info("MC: run_single: built `distance` at shape %s.", distance.shape)

    run_param = (sep, ecc, v_mag, impact_parameter, theta, phi, psi, distance, true_anomalies, "MC")
    logger.info("MC: run_single: built `run_param` at length %s.", len(run_param))
    # try:
    logger.info("MC: run_single: run_6body try start.")
    frames, outcome = run_6_body_simulation(*run_param)
    logger.info("MC: run_single: run_6body finished.")
    return outcome, w

    # except Exception:
    #     logger.warning("MC: run_single: run_6body at exception.")
    #     return "simulation_failed", w


@dataclass
class MonteCarloResult:
    samples: np.ndarray  # shape (n_samples, 19)
    param_names: list  # list of 19 strings
    distances: np.ndarray  # shape (n_samples, 2)
    weights: np.ndarray  # shape (n_samples,)
    all_star_outcomes: np.ndarray  # structured array for all stars with >=1 collision
    all_star_weights: np.ndarray  # weight for each star (same as sample weight)
    sample_ids: np.ndarray  # mapping from each star back to its MC sample row
    weighted_counts: np.ndarray  # array of total weights per outcome
    probabilities: np.ndarray  # normalized probabilities per outcome
    unique_outcomes: np.ndarray  # outcome labels


def monte_carlo_19D(n_samples, n_cores=None, verbose=True) -> MonteCarloResult:
    if n_cores is None:
        n_cores = cpu_count()
        logger.info("MC: main: `n_cores` not found, default to `cpu_count()` at %d.", n_cores)

    # Array-based samples
    samples, param_names, distances, weights = sample_19D_lhs(n_samples)
    logger.info("MC: main: `samples, param_names, distances, weights` obtained from `lhs`.")
    pool_args = [(samples[i], distances[i], weights[i]) for i in range(n_samples)]
    logger.info("MC: main: `pool_args` built with length %s.", len(pool_args))

    # Run simulations in parallel
    results = []
    logger.info("MC: main: init `results` at `[]`.")
    with Pool(processes=n_cores) as pool:
        for summary_outcome, w in tqdm(
            pool.imap_unordered(_run_single_simulation, pool_args),
            total=n_samples,
            disable=not verbose,
        ):
            results.append((summary_outcome, w))
            logger.info("MC: main: `results` appended with `(summary_outcome, w)`.")

    # Flatten stars with >=1 collision and record which MC sample they came from
    all_star_outcomes, all_star_weights, sample_ids = [], [], []
    logger.info("MC: main: init `all_star_outcomes, all_star_weights, sample_ids` at `[], [], []`.")
    for sample_idx, (summary_outcome, w) in enumerate(results):
        if summary_outcome == "simulation_failed":
            logger.info("MC: main: continuing at `simulation failed`.")
            continue
        stars = [s for s in summary_outcome if s["collisions"] >= 1]
        for star in stars:
            all_star_outcomes.append(
                (
                    star["star_key"],
                    star["collisions"],
                    star["n_companions"],
                    star["mass_Msun"],
                    star["outcome"],
                )
            )
            all_star_weights.append(w)
            sample_ids.append(sample_idx)  # << record MC sample index
            logger.info(
                "MC: main: results appended to `all_star_outcomes, all_star_weights, sample_ids`."
            )

    dtype = [
        ('star_key', 'uint64'),
        ('collisions', 'int32'),
        ('n_companions', 'int32'),
        ('mass_Msun', 'float64'),
        ('outcome', 'U32'),
    ]
    all_star_outcomes_array = np.array(all_star_outcomes, dtype=dtype)
    logger.info("MC: main: build `all_star_outcomes_array`.")
    all_star_weights_array = np.array(all_star_weights, dtype=float)
    logger.info("MC: main: build `all_star_weights_array`.")
    sample_ids_array = np.array(sample_ids, dtype=int)
    logger.info("MC: main: build `ids_array`.")

    # Compute weighted counts and probabilities
    unique_outcomes, inverse_idx = np.unique(
        all_star_outcomes_array['outcome'], return_inverse=True
    )
    weighted_counts = np.zeros(len(unique_outcomes))
    for idx, w in zip(inverse_idx, all_star_weights_array):
        weighted_counts[idx] += w
    probabilities = weighted_counts / weighted_counts.sum()
    logger.info("MC: main: all primary tasks completed.")

    return MonteCarloResult(
        samples=samples,
        param_names=param_names,
        distances=distances,
        weights=weights,
        all_star_outcomes=all_star_outcomes_array,
        all_star_weights=all_star_weights_array,
        sample_ids=sample_ids_array,
        weighted_counts=weighted_counts,
        probabilities=probabilities,
        unique_outcomes=unique_outcomes,
    )
