import corner
import matplotlib.pyplot as plt
import numpy as np

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
        "ecc",
        "sep",
        "v_mag",
        "impact_parameter",
        "theta",
        "phi",
        "psi",
        "true_anomalies",
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
        bins=25,
    )

    plt.show()


def plot_corner_marginalized(
    samples, results, outcome_name="creative_ionized", param_subset=None
):
    """
    Generate a corner plot for a subset of parameters, marginalizing over the others.

    Parameters
    ----------
    samples : list of dicts
        Output of sample_19D_lhs() (old format)
    results : list of (label, weight)
        Full simulation outcomes and weights
    outcome_name : str
        Outcome category to filter (e.g., "creative_ionized")
    param_subset : list of str
        List of parameter names to include in the plot (e.g., ["ecc", "sep", "v_mag"])
        If None, all parameters are included.
    """
    # ---- 1. Collect matching sample indices ----
    indices = [i for i, (label, _) in enumerate(results) if label == outcome_name]

    if len(indices) == 0:
        print(f"No samples with outcome '{outcome_name}'. Cannot make corner plot.")
        return

    # Default to all parameters
    if param_subset is None:
        param_subset = [
            "ecc",
            "sep",
            "v_mag",
            "impact_parameter",
            "theta",
            "phi",
            "psi",
            "true_anomalies",
        ]

    # ---- 2. Build matrix of parameters ----
    data = []
    weights = []

    for idx in indices:
        s = samples[idx]
        w = results[idx][1]  # weight
        row = []
        for param in param_subset:
            row.extend(s[param])
        data.append(row)
        weights.append(w)

    data = np.array(data)
    weights = np.array(weights)

    # ---- 3. Build axis labels ----
    axis_labels = []
    for param in param_subset:
        n = len(samples[0][param])
        for i in range(n):
            axis_labels.append(f"{param}_{i}")

    # ---- 4. Plot weighted corner plot ----
    fig = corner.corner(
        data,
        labels=axis_labels,
        show_titles=True,
        title_fmt=".2f",
        bins=25,
        weights=weights,
    )

    plt.show()
