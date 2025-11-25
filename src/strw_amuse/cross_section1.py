
# Imports 
from multiprocessing import Pool, cpu_count
from collections import Counter
from tqdm import tqdm
import corner
import matplotlib.pyplot as plt
import numpy as np

from amuse.units import units
from src.strw_amuse.run_simulation import run_6_body_simulation


# -------------------------
# Sampling initial conditions
# ------------------------- 
def sample_19D_lhs(n_samples, rng=None):
    """
    Generate stratified Latin Hypercube samples in the fixed 19D parameter space
    for 3 binaries + 2 incoming binaries encounters.
    Returns: list of sample dictionaries, each with keys:
        ecc, sep, v_mag, impact_parameter, theta, phi, psi, true_anomalies, distance, weight
    """

    if rng is None:
        rng = np.random.default_rng()

    samples = []

    # --- Define the parameter ranges ---
    param_ranges = {
        "ecc":              [(0.0, 0.99)] * 3,   # 3 binaries
        "sep":              [(2.0, 50.0)] * 3,   # AU
        "v_mag":            [(0.1, 1.0)] * 2,    # incoming binaries
        "impact_parameter": [(0.0, 5.0)] * 2,    # AU
        "theta":            [(0.0, np.pi/2)] * 2,
        "phi":              [(0.0, 2*np.pi)] * 2,
        "psi":              [(0.0, 2*np.pi)] * 2,
        "true_anomalies":   [(0.0, 2*np.pi)] * 3
    }

    # Flatten param ranges
    param_names, param_bounds = [], []
    for k, bounds in param_ranges.items():
        for i, b in enumerate(bounds):
            param_names.append(f"{k}_{i}")
            param_bounds.append(b)

    n_params = len(param_names)  # should be 19

    # --- Latin Hypercube: stratify each dimension ---
    lhs = rng.uniform(size=(n_samples, n_params))
    for j in range(n_params):
        lhs[:, j] = (lhs[:, j] + np.arange(n_samples)) / n_samples

    # --- Scale to parameter ranges and assemble sample dicts ---
    for i in range(n_samples):
        sample_dict = {k: [] for k in param_ranges.keys()}  # grouped by type
        for j, pname in enumerate(param_names):
            low, high = param_bounds[j]
            val = low + lhs[i, j] * (high - low)
            base_name = "_".join(pname.split("_")[:-1])   # <--- FIXED
            sample_dict[base_name].append(val)
        # fixed outer distances
        sample_dict["distance"] = [50.0, 50.0]
        sample_dict["weight"] = 1.0
        samples.append(sample_dict)

    return samples

# -------------------------
# Helper fixing run simulation outcomes: use that or fix outcome format in the main simulation function
# ------------------------- 
def sanitize_outcome(outcome):
    """
    Convert any outcome from run_6_body_simulation into a pickle-safe label.
    Detect creative_ionized remnants even inside multi-collision outcomes.
    """
    # outcome = (label, info)
    if isinstance(outcome, tuple) and len(outcome) == 2:
        label, info = outcome

        # Single collision categories are already strings
        if label != "multiple_collisions":
            return str(label), None

        # MULTIPLE COLLISIONS: inspect info list
        if isinstance(info, list):
            # check if any remnant is creative_ionized
            for rem in info:
                if rem.get("type") == "creative_ionized":
                    return "creative_ionized", None
            # otherwise check if any are bound massive
            for rem in info:
                if rem.get("type") == "creative_bound":
                    return "creative_bound", None

            # fallback
            return "multiple_collisions", None

    if isinstance(outcome, list):
        return "multiple_collisions", None

    return "unknown", None

# -------------------------
# Single Simulation
# ------------------------- 
def _run_single_simulation(sample):
    w = sample["weight"]
    try:
        frames, outcome = run_6_body_simulation(
            sample["sep"],
            sample["true_anomalies"],
            sample["ecc"],
            sample["theta"],
            sample["phi"],
            sample["v_mag"],
            sample["impact_parameter"],
            sample["psi"],
            sample["distance"],
            run_label="MC"
        )

        safe_label, _ = sanitize_outcome(outcome)
        return safe_label, w

    except Exception as e:
        return "simulation_failed", w

# -------------------------
# Monte Carlo Simulations
# ------------------------- 
def monte_carlo_19D(n_samples, n_cores=None, verbose=True):
    """
    Run n_samples 6-body simulations using Latin Hypercube sampling in 19D parameter space.
    Executes in parallel using n_cores.
    Returns weighted counts and probabilities of outcomes.
    """
    if n_cores is None:
        n_cores = cpu_count()

    samples = sample_19D_lhs(n_samples)

    # --- Run simulations in parallel ---
    results = []
    with Pool(processes=n_cores) as pool:
        for outcome_label, weight in tqdm(pool.imap_unordered(_run_single_simulation, samples),
                                          total=n_samples, disable=not verbose):
            results.append((outcome_label, weight))

    # --- Weighted counts ---
    weighted_counts = Counter()
    for label, w in results:
        weighted_counts[label] += w

    total_weight = sum(weighted_counts.values()) - weighted_counts.get("simulation_failed", 0)

    # --- Probabilities ---
    probabilities = {k: v / total_weight for k, v in weighted_counts.items() if k != "simulation_failed"}

    return dict(
        n_samples=n_samples,
        samples=samples,
        results=results,                 # <--- add this
        weighted_counts=weighted_counts,
        probabilities=probabilities
    )

# -------------------------
# Corner Plot 
# ------------------------- 

def plot_corner_for_outcome(samples, results, outcome_name="creative_ionized"):
    """
    Generate a corner plot for all samples that produced the given outcome_name.
    
    Parameters
    ----------
    samples : list of dicts
        Output of sample_19D_lhs()
    results : list of (label, weight)
        Output collected from the parallel pool
    outcome_name : str
        Outcome category to filter (e.g., "creative_ionized")
    """

    # ---- 1. Collect matching sample indices ----
    indices = [i for i, (label, _) in enumerate(results) if label == outcome_name]

    if len(indices) == 0:
        print(f"No samples with outcome '{outcome_name}'. Cannot make corner plot.")
        return

    # ---- 2. Build matrix of parameters ----
    data = []
    labels = []

    # build list of parameter names in stable order
    param_order = [
        "ecc", "sep", "v_mag", "impact_parameter",
        "theta", "phi", "psi", "true_anomalies"
    ]
    # count lengths = 3, 3, 2, 2, 2, 2, 2, 3

    # Construct readable axis labels
    axis_labels = []
    for k in param_order:
        for i in range(len(samples[0][k])):
            axis_labels.append(f"{k}_{i}")

    # Fill matrix
    for idx in indices:
        s = samples[idx]
        row = []
        for k in param_order:
            row.extend(list(s[k]))
        data.append(row)

    data = np.array(data)

    # ---- 3. Plot ----
    fig = corner.corner(
        data,
        labels=axis_labels,
        show_titles=True,
        title_fmt=".2f",
        quantiles=[0.16, 0.5, 0.84],
        bins=25
    )

    plt.show()
