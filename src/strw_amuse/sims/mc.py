"""
Monte Carlo utilities for cross section calculation.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .run_sim import run_6_body_simulation
from ..core import sampler

logger = logging.getLogger(__name__)


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


def _run_single_simulation(args):
    sample_row, distance_row, w = args

    ecc = sample_row[0:3]
    sep = sample_row[3:6]
    v_mag = sample_row[6:8]
    impact = sample_row[8:10]
    theta = sample_row[10:12]
    phi = sample_row[12:14]
    psi = sample_row[14:16]
    anomalies = sample_row[16:19]

    try:
        _, outcome = run_6_body_simulation(
            sep=sep,
            ecc=ecc,
            v_mag=v_mag,
            impact_parameter=impact,
            theta=theta,
            phi=phi,
            psi=psi,
            distance=distance_row,
            true_anomalies=anomalies,
            run_label="MC",
        )
        return outcome, w
    except Exception:
        return "simulation_failed", w


def monte_carlo_19D(n_samples, verbose=True):
    """
    Monte Carlo 19D simulation for a single job segment using `sampler.gen_nd_samples`.

    Returns a `MonteCarloResult` dataclass with flattened star outcomes and mapping to sample rows.
    """
    # --------------------------
    # Generate samples
    # --------------------------
    _, samples = sampler.gen_nd_samples(n_samples=n_samples)

    # Since your old function returned param_names, distances, weights
    # we reconstruct them here
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
    param_names = [
        f"{label}_{i}" for label, count in zip(param_labels, param_counts) for i in range(count)
    ]
    distances = np.full((n_samples, 2), 100.0)
    weights = np.ones(n_samples)

    # --------------------------
    # Prepare args for simulation
    # --------------------------
    pool_args = list(zip(samples, distances, weights))
    results = []

    # Sequential execution (can parallelize with Pool later)
    for args in tqdm(pool_args, total=n_samples, disable=not verbose):
        outcome, w = _run_single_simulation(args)
        results.append((outcome, w))

    # ----------------------------------------
    # Flatten stars with >=1 collision and store sample mapping
    # ----------------------------------------
    flat_data = []
    flat_weights = []
    flat_ids = []

    for sample_idx, (outcome, w) in enumerate(results):
        if outcome == "simulation_failed":
            continue
        stars = [s for s in outcome if s["collisions"] >= 1]
        for s in stars:
            flat_data.append(
                (
                    s["star_key"],
                    s["collisions"],
                    s["n_companions"],
                    s["mass_Msun"],
                    s["outcome"],
                )
            )
            flat_weights.append(w)
            flat_ids.append(sample_idx)

    dtype = [
        ("star_key", "uint64"),
        ("collisions", "int32"),
        ("n_companions", "int32"),
        ("mass_Msun", "float64"),
        ("outcome", "U32"),
    ]

    all_star_outcomes = np.array(flat_data, dtype=dtype)
    all_star_weights = np.array(flat_weights)
    sample_ids = np.array(flat_ids)

    # ----------------------------------------
    # Compute weighted counts and probabilities
    # ----------------------------------------
    unique_outcomes, inv_idx = np.unique(all_star_outcomes["outcome"], return_inverse=True)
    weighted_counts = np.bincount(inv_idx, weights=all_star_weights)
    probabilities = weighted_counts / weighted_counts.sum()

    return MonteCarloResult(
        samples=samples,
        param_names=param_names,
        distances=distances,
        weights=weights,
        all_star_outcomes=all_star_outcomes,
        all_star_weights=all_star_weights,
        sample_ids=sample_ids,
        weighted_counts=weighted_counts,
        probabilities=probabilities,
        unique_outcomes=unique_outcomes,
    )
