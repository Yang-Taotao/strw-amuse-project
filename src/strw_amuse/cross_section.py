# Imports
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from src.strw_amuse.run_simulation import run_6_body_simulation


# -------------------------
# Sampling initial conditions
# -------------------------
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
        rng = np.random.default_rng(seed=42)

    # --- Define independent parameter counts ---
    param_counts = np.array(
        [3, 3, 2, 2, 2, 2, 2, 3]
    )  # ecc, sep, v_mag, impact, theta, phi, psi, anomalies
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
    upper_bounds = np.array([0.99, 50.0, 1.0, 5.0, np.pi / 2, 2 * np.pi, 2 * np.pi, 2 * np.pi])

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

    # Fixed distances and weights
    distances = np.full((n_samples, 2), 150.0)
    weights = np.ones(n_samples)

    return samples, param_names, distances, weights


# -------------------------
# Single Simulation
# -------------------------
def _run_single_simulation(args):
    """
    Runs a single simulation given array slices.
    args = (sample_row, distances_row, weight)
    """
    sample_row, distances_row, w = args

    # Column indices based on param_names in sample_19D_lhs
    # param_names = ["ecc_0","ecc_1","ecc_2", "sep_0","sep_1","sep_2",
    #                "v_mag_0","v_mag_1", "impact_parameter_0","impact_parameter_1",
    #                "theta_0","theta_1", "phi_0","phi_1", "psi_0","psi_1",
    #                "true_anomalies_0","true_anomalies_1","true_anomalies_2"]

    ecc = sample_row[0:3]
    sep = sample_row[3:6]
    v_mag = sample_row[6:8]
    impact_parameter = sample_row[8:10]
    theta = sample_row[10:12]
    phi = sample_row[12:14]
    psi = sample_row[14:16]
    true_anomalies = sample_row[16:19]
    distance = distances_row

    try:
        frames, outcome = run_6_body_simulation(
            sep,
            true_anomalies,
            ecc,
            theta,
            phi,
            v_mag,
            impact_parameter,
            psi,
            distance,
            run_label="MC",
        )

        return outcome, w

    except Exception as e:
        return "simulation_failed", w


# -------------------------
# Monte Carlo Simulations
# -------------------------


@dataclass
class MonteCarloResult:
    samples: np.ndarray  # shape (n_samples, 19)
    param_names: list  # list of 19 strings
    distances: np.ndarray  # shape (n_samples, 2)
    weights: np.ndarray  # shape (n_samples,)
    all_star_outcomes: np.ndarray  # structured array for all stars with >=1 collision
    weighted_counts: np.ndarray  # array of total weights per outcome
    probabilities: np.ndarray  # normalized probabilities per outcome
    unique_outcomes: np.ndarray  # outcome labels


def monte_carlo_19D(n_samples, n_cores=None, verbose=True) -> MonteCarloResult:
    if n_cores is None:
        n_cores = cpu_count()

    # Array-based samples
    samples, param_names, distances, weights = sample_19D_lhs(n_samples)
    pool_args = [(samples[i], distances[i], weights[i]) for i in range(n_samples)]

    # Run simulations in parallel
    results = []
    with Pool(processes=n_cores) as pool:
        for summary_outcome, w in tqdm(
            pool.imap_unordered(_run_single_simulation, pool_args),
            total=n_samples,
            disable=not verbose,
        ):
            results.append((summary_outcome, w))

    # Flatten stars with >=1 collision
    all_star_outcomes, all_star_weights = [], []
    for summary_outcome, w in results:
        if summary_outcome == "simulation_failed":
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

    dtype = [
        ("star_key", "uint64"),
        ("collisions", "int32"),
        ("n_companions", "int32"),
        ("mass_Msun", "float64"),
        ("outcome", "U32"),
    ]
    all_star_outcomes_array = np.array(all_star_outcomes, dtype=dtype)
    all_star_weights_array = np.array(all_star_weights, dtype=float)

    # Compute weighted counts and probabilities
    unique_outcomes, inverse_idx = np.unique(
        all_star_outcomes_array["outcome"], return_inverse=True
    )
    weighted_counts = np.zeros(len(unique_outcomes))
    for idx, w in zip(inverse_idx, all_star_weights_array):
        weighted_counts[idx] += w
    probabilities = weighted_counts / weighted_counts.sum()

    return MonteCarloResult(
        samples=samples,
        param_names=param_names,
        distances=distances,
        weights=weights,
        all_star_outcomes=all_star_outcomes_array,
        weighted_counts=weighted_counts,
        probabilities=probabilities,
        unique_outcomes=unique_outcomes,
    )
