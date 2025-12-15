"""
Plotting utilities for AMUSE simulation.
"""

import logging
import os

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.colors as mcolors
import numpy as np
from amuse.units import units
from matplotlib.animation import FuncAnimation, PillowWriter

from ..core import sampler
from ..utils.config import (
    BOUNDS,
    N_DIMS,
    N_SAMPLES,
    OUTPUT_DIR_GIF,
    OUTPUT_DIR_IMG,
    OUTPUT_DIR_SAMPLER,
)

logger = logging.getLogger(__name__)

mpl.rcParams["text.usetex"] = False
plt.style.use(['science', 'nature', 'no-latex'])


def plot_gif(frames, run_label="test", massive_threshold=70.0):
    """
    Visualize AMUSE simulation frames and produce a GIF.

    For the final frame, finds all stars with mass >= massive_threshold M☉.
    Produces one subplot per massive star.
    Each subplot tracks the corresponding star through all frames.
    Axis limits are fixed to [-100, 100] AU.
    Color-coded by stellar mass with a shared colorbar.
    """

    if len(frames) == 0:
        logger.warning("No frames provided to `plot_gif()`.")
        return None

    output_dir = OUTPUT_DIR_GIF
    os.makedirs(output_dir, exist_ok=True)

    # --- Identify most massive stars in the final frame ---
    final_frame = frames[-1]
    final_masses = np.array([p.mass.value_in(units.MSun) for p in final_frame])
    massive_indices = np.where(final_masses >= massive_threshold)[0]
    if len(massive_indices) == 0:
        massive_indices = [np.argmax(final_masses)]
    n_massive = len(massive_indices)
    logger.info("Tracking %d massive stars in final frame", n_massive)

    # --- Extract positions of tracked stars in each frame ---
    tracked_positions = np.zeros((len(frames), n_massive, 2))
    for f_idx, frame in enumerate(frames):
        for idx_i, star_idx in enumerate(massive_indices):
            if star_idx < len(frame):
                p = frame[star_idx]
            else:
                final_pos = np.array(
                    [
                        final_frame[star_idx].x.value_in(units.AU),
                        final_frame[star_idx].y.value_in(units.AU),
                    ]
                )
                xs = np.array([p.x.value_in(units.AU) for p in frame])
                ys = np.array([p.y.value_in(units.AU) for p in frame])
                dists = np.hypot(xs - final_pos[0], ys - final_pos[1])
                nearest_idx = int(np.argmin(dists))
                p = frame[nearest_idx]

            tracked_positions[f_idx, idx_i, 0] = p.x.value_in(units.AU)
            tracked_positions[f_idx, idx_i, 1] = p.y.value_in(units.AU)

    # --- Gather all masses for color mapping ---
    all_masses = np.array([p.mass.value_in(units.MSun) for f in frames for p in f])
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=all_masses.min(), vmax=all_masses.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_to_color(m):
        return cmap(norm(m))

    # --- Figure and subplots setup ---
    fig, axes = plt.subplots(1, n_massive, figsize=(8 * n_massive, 8))
    if n_massive == 1:
        axes = [axes]

    sc_list = []
    time_text_list = []
    for ax in axes:
        sc = ax.scatter([], [], s=[])
        sc_list.append(sc)
        t_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            color="black",
        )
        time_text_list.append(t_text)
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    # --- Add a shared colorbar ---
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"Mass [M$_\odot$]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # --- Initialization ---
    def init():
        for sc, t_text in zip(sc_list, time_text_list):
            sc.set_offsets(np.empty((0, 2)))
            t_text.set_text("")
        return sc_list + time_text_list

    # --- Update function ---
    def update(frame_idx):
        frame = frames[frame_idx]
        masses_frame = np.array([p.mass.value_in(units.MSun) for p in frame])
        x_frame = np.array([p.x.value_in(units.AU) for p in frame])
        y_frame = np.array([p.y.value_in(units.AU) for p in frame])

        for ax_idx, star_idx in enumerate(massive_indices):
            center = tracked_positions[frame_idx, ax_idx]
            x_rel = x_frame - center[0]
            y_rel = y_frame - center[1]
            sizes = np.clip(masses_frame * 2, 10, 500)
            colors = [mass_to_color(m) for m in masses_frame]

            sc_list[ax_idx].set_offsets(np.c_[x_rel, y_rel])
            sc_list[ax_idx].set_sizes(sizes)
            sc_list[ax_idx].set_color(colors)
            time_text_list[ax_idx].set_text(f"t = {frame_idx*0.1:.1f} yr")

        return sc_list + time_text_list

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=50,
        blit=False,
        repeat=False,
    )

    gif_filename = os.path.join(OUTPUT_DIR_GIF, f"encounter_evolution_{run_label}.gif")
    writer = PillowWriter(fps=10)
    ani.save(gif_filename, writer=writer)
    plt.close(fig)
    logger.info("GIF saved as %s", gif_filename)


def plot_trajectory(frames, run_label="test"):
    """
    Spaghetti plot of final frame stars and their trajectories over time.
    Centers on the most massive star in the final frame.
    """
    if len(frames) == 0:
        logger.warning("No frames provided for `plot_trajectory()`.")
        return

    output_dir = OUTPUT_DIR_IMG
    os.makedirs(output_dir, exist_ok=True)

    final = frames[-1]

    # ---------------------------------------------------------
    # Build colormap for masses
    # ---------------------------------------------------------
    all_masses = np.array([p.mass.value_in(units.MSun) for f in frames for p in f])
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=all_masses.min(), vmax=all_masses.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_color(m):
        return cmap(norm(m))

    # ---------------------------------------------------------
    # Track most massive star for centering
    # ---------------------------------------------------------
    masses_final = np.array([p.mass.value_in(units.MSun) for p in final])
    idx_track = np.argmax(masses_final)

    track_pos = np.array(
        [
            (
                [fr[idx_track].x.value_in(units.AU), fr[idx_track].y.value_in(units.AU)]
                if idx_track < len(fr)
                else [0, 0]
            )
            for fr in frames
        ]
    )
    cx_final, cy_final = track_pos[-1]

    # ---------------------------------------------------------
    # Prepare figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title(f"Spaghetti plots: {run_label}")
    plt.colorbar(sm, ax=ax, label=r"Mass [$M_{\odot}$]")

    # ---------------------------------------------------------
    # Helper to scatter stars
    # ---------------------------------------------------------
    def scatter_stars(ax, frame, center=(0, 0)):
        x = np.array([p.x.value_in(units.AU) for p in frame]) - center[0]
        y = np.array([p.y.value_in(units.AU) for p in frame]) - center[1]
        m = np.array([p.mass.value_in(units.MSun) for p in frame])
        sizes = np.clip(m * 3, 8, 500)
        colors = [mass_color(mi) for mi in m]
        ax.scatter(x, y, s=sizes, c=colors)
        return x, y

    # ---------------------------------------------------------
    # Scatter final positions
    # ---------------------------------------------------------
    x_rel, y_rel = scatter_stars(ax, final, center=(cx_final, cy_final))

    # ---------------------------------------------------------
    # Plot full trajectories
    # ---------------------------------------------------------
    for idx_final in range(len(final)):
        traj_x = []
        traj_y = []

        for fr, (cx_tr, cy_tr) in zip(frames, track_pos):
            if idx_final >= len(fr):
                break
            traj_x.append(fr[idx_final].x.value_in(units.AU) - cx_tr)
            traj_y.append(fr[idx_final].y.value_in(units.AU) - cy_tr)

        traj_x = np.array(traj_x)
        traj_y = np.array(traj_y)

        if len(traj_x) > 1:
            ax.plot(traj_x, traj_y, alpha=0.6)

    # Adaptive limits
    dx = max(50, (x_rel.max() - x_rel.min()) * 1.3)
    dy = max(50, (y_rel.max() - y_rel.min()) * 1.3)
    ax.set_xlim(-dx, dx)
    ax.set_ylim(-dy, dy)

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    png_filename = os.path.join(output_dir, f"Spaghetti_plot_{run_label}.png")
    fig.savefig(png_filename)
    plt.close(fig)
    logger.info("Spaghetti plot saved as %s", png_filename)


def sampler_nd_coverage_plot(
    samples_np: np.ndarray,
    samples_sp: np.ndarray,
    n_samples: int = N_SAMPLES,
    n_dims: int = N_DIMS,
    bounds: np.ndarray = BOUNDS,
    save_dir: str = OUTPUT_DIR_SAMPLER,
) -> None:
    """
    General plotter for sample coverage visualization.

    Args:
        samples_np (np.ndarray): Uniform samples from `numpy.random.uniform()`.
        samples_sp (np.ndarray): LHS samples from `scipy.stats.qmc.LatinHyperCube()`
        n_samples (int, optional): Defaults to N_SAMPLES.
        n_dims (int, optional): Defaults to N_DIMS.
        bounds (np.ndarray, optional): Defaults to BOUNDS.
        save_dir (str, optional): Defaults to OUTPUT_DIR_SAMPLER.
    """

    # local repo
    low, high = sampler.nd_bounds(bounds)
    coverage_np = sampler.nd_coverage(samples_np)
    coverage_sp = sampler.nd_coverage(samples_sp)
    cmap = plt.get_cmap("viridis")
    np_color = cmap(0.7)
    sp_color = cmap(0.3)

    # fig init
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Uniform coverage
    bars_np = ax1.bar(range(n_dims), coverage_np, color=np_color, alpha=0.7, label='Uniform')
    ax1.axhline(1.0, color='red', ls='--', lw=2, label='Perfect 100%')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Range Coverage')
    ax1.set_title('Uniform: Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: LHS coverage
    bars_sp = ax2.bar(range(n_dims), coverage_sp, color=sp_color, alpha=0.8, label='LHS')
    ax2.axhline(1.0, color='red', ls='--', lw=2, label='Perfect 100%')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Range Coverage')
    ax2.set_title('LHS: Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1d param converage comparison
    for i, cov in enumerate(coverage_sp):
        if cov > 0.99:
            ax2.text(i, cov + 0.01, '✓', ha='center', va='bottom', fontsize=14, color='red')

    fig.suptitle('Parameter Space Coverage Comparison', fontsize=16)
    plt.tight_layout()
    fig.savefig(f'{save_dir}/1d_param_coverage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # corner
    fig_left = plt.figure(figsize=(10, 10))
    corner.corner(
        samples_np,
        fig=fig_left,
        color=np_color,
        alpha=0.7,
        labels=[f'p{i}' for i in range(n_dims)],
        range=[(low[i], high[i]) for i in range(n_dims)],
        quantiles=[0.25, 0.5, 0.75],
        smooth=0.05,
        plot_datapoints=False,
        plot_density=False,
    )

    fig_left.suptitle(f'Uniform Corner Plot ({n_samples} samples)', fontsize=18)
    plt.savefig(f'{save_dir}/uniform.png', dpi=300, bbox_inches='tight')
    plt.close(fig_left)

    fig_right = plt.figure(figsize=(10, 10))
    corner.corner(
        samples_sp,
        fig=fig_right,
        color=sp_color,
        alpha=0.7,
        labels=[f'p{i}' for i in range(n_dims)],
        range=[(low[i], high[i]) for i in range(n_dims)],
        quantiles=[0.25, 0.5, 0.75],
        smooth=0.05,
        plot_datapoints=False,
    )

    fig_right.suptitle(f'LHS Corner Plot ({n_samples} samples)', fontsize=18)
    plt.savefig(f'{save_dir}/lhs.png', dpi=300, bbox_inches='tight')
    plt.close(fig_right)

    # Summary stats
    print("\n" + "=" * 50)
    print("1D COVERAGE SUMMARY")
    print("=" * 50)
    print(f"Uniform sampling avg: {np.mean(coverage_np):.1%} ± {np.std(coverage_np):.1%}")
    print(f"LHS     sampling avg: {np.mean(coverage_sp):.1%} ± {np.std(coverage_sp):.1%}")
    print(f"LHS          perfect: {sum(1 for x in coverage_sp if x > 0.99)}/19")
    print(f"Uniform      perfect: {sum(1 for x in coverage_np if x > 0.99)}/19")
    print("=" * 50)


def plot_cross_section(
    mc_result, outcome_name, param_groups, n_bins=20, b_max_dict=None, save_dir=OUTPUT_DIR_IMG
):
    """
    Plot differential cross-section vs parameters for a given outcome.

    Parameters
    ----------
    mc_result : MonteCarloResult
        Monte Carlo result object.
    outcome_name : str
        Outcome to compute cross-section for (e.g., "Creative_ionized").
    param_groups : dict
        Keys are plot titles / labels (e.g., "ecc")
            values are lists of parameter names to marginalize over.
        Example: {"ecc": ["ecc_0","ecc_1","ecc_2"], "sep": ["sep_0","sep_1","sep_2"]}
    n_bins : int
        Number of bins per parameter group.
    b_max_dict : dict or None
        Optional dictionary specifying max bin values per group.
    show : bool
        Whether to show the plot immediately.
    """
    # Mask stars with desired outcome
    outcome_mask = mc_result.all_star_outcomes['outcome'] == outcome_name
    if not np.any(outcome_mask):
        print(f"No stars produced outcome '{outcome_name}'")
        return

    weights = mc_result.all_star_weights[outcome_mask]
    sample_ids = mc_result.sample_ids[outcome_mask]

    n_plots = len(param_groups)
    n_col = 2
    n_row = (n_plots + n_col) // n_col

    cmap = plt.get_cmap("viridis")
    band_color = cmap(0.7)
    point_color = cmap(0.3)

    plt.figure(figsize=(8 * n_col, 6 * n_row))

    for i, (group_name, param_names) in enumerate(param_groups.items(), start=1):
        # Collect all parameter values for the group
        param_values_list = []
        for pname in param_names:
            col_idx = [j for j, name in enumerate(mc_result.param_names) if name == pname]
            if len(col_idx) == 0:
                raise ValueError(f"Parameter {pname} not found in samples")
            param_values = mc_result.samples[sample_ids, col_idx[0]]
            param_values_list.append(param_values)

        # Flatten for marginalization
        all_values = np.hstack([v[:, None] for v in param_values_list]).flatten()
        all_weights = np.hstack([weights[:, None] for _ in param_values_list]).flatten()

        # Define bins
        if b_max_dict is None or group_name not in b_max_dict:
            b_max = np.max(all_values) * 1.05
        else:
            b_max = b_max_dict[group_name]
        b_bins = np.linspace(0, b_max, n_bins + 1)
        b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])

        sigma_binned = np.zeros(n_bins)
        sigma_err = np.zeros(n_bins)

        for j in range(n_bins):
            mask = (all_values >= b_bins[j]) & (all_values < b_bins[j + 1])
            w_bin = all_weights[mask]
            if len(w_bin) == 0:
                continue
            area = np.pi * (b_bins[j + 1] ** 2 - b_bins[j] ** 2)  # keep same definition
            sigma_binned[j] = area * np.sum(w_bin)
            sigma_err[j] = area * np.sqrt(np.sum(w_bin**2))

        # Plot
        ax = plt.subplot(n_row, n_col, i)

        # 1) Error band (semi-transparent)
        upper = sigma_binned + sigma_err
        lower = sigma_binned - sigma_err
        lower = np.clip(lower, 0.0, None)  # avoid negative cross-sections

        ax.fill_between(
            b_centers,
            lower,
            upper,
            color=band_color,
            alpha=0.5,
            linewidth=0,
            zorder=1,
            label="uncertainty",
        )

        ax.plot(b_centers, upper, color=band_color, alpha=0.5, linewidth=0.5)
        ax.plot(b_centers, lower, color=band_color, alpha=0.5, linewidth=0.5)

        # 2) Points only (no connecting line, smaller markers)
        ax.scatter(
            b_centers,
            sigma_binned,
            s=24,
            color=point_color,
            edgecolors="none",
            zorder=3,
            label="binned cross-section",
        )

        ax.set_xlabel(group_name, fontsize=14)
        ax.set_ylabel(r"$\sigma_\mathrm{ionized}\left(\mathrm{AU}^2\right)$", fontsize=14)

        ax.set_yscale("log")

        # Optional: show legend only once or on each subplot
        if i == 1:
            ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/cross_section.png', dpi=300)
    plt.close()
    logger.info("VIS: `plot_cross_section()` finished.")


def corner_for_outcome(
    mc_result,
    outcome_name,
    param_subset=None,
    bins=20,
    title_fmt=".2f",
    show_titles=True,
    save_dir=OUTPUT_DIR_IMG,
):
    """
    Generate a corner plot for samples that produced a given outcome.

    Parameters
    ----------
    mc_result : MonteCarloResult
        The object returned from `monte_carlo_19D`.
    outcome_name : str
        The outcome to filter on (e.g., "Creative_ionized").
    param_subset : list of str, optional
        Subset of parameter names to include in the corner plot. Default is all.
    bins : int
        Number of bins for histograms.
    title_fmt : str
        Formatting for histogram titles.
    show_titles : bool
        Whether to show titles on each subplot.
    """

    # ---- 1. Identify stars with the desired outcome ----
    outcome_mask = mc_result.all_star_outcomes['outcome'] == outcome_name
    if not np.any(outcome_mask):
        print(f"No samples produced outcome '{outcome_name}'")
        return

    # Map stars back to unique MC sample rows
    sample_rows = np.unique(mc_result.sample_ids[outcome_mask])
    n_samples_plot = len(sample_rows)

    # ---- 2. Select parameters to plot ----
    if param_subset is None:
        data_to_plot = mc_result.samples[sample_rows, :]
        labels = mc_result.param_names
    else:
        # Find indices of selected parameters
        indices = [mc_result.param_names.index(p) for p in param_subset]
        data_to_plot = mc_result.samples[sample_rows][:, indices]
        labels = param_subset

    # ---- 3. Check if sufficient data are present ----
    n_dims = data_to_plot.shape[1]
    if n_samples_plot < max(10, 2 * n_dims):
        logger.info("VIS: Insufficient data, `corner_for_outcome()` skipped")
        return None

    # ---- 4. Make corner plot ----
    cmap = plt.get_cmap("viridis")
    color = cmap(0.3)

    fig = plt.figure(figsize=(10, 10))

    label_kwargs = {"fontsize": 16}
    title_kwargs = {"fontsize": 18}
    hist_kwargs = {"color": color}

    fig = corner.corner(
        data_to_plot,
        labels=labels,
        bins=bins,
        show_titles=show_titles,
        title_fmt=title_fmt,
        color=color,
        plot_datapoints=True,
        plot_contours=True,
        plot_density=True,
        fill_contours=True,
        label_kwargs=label_kwargs,
        title_kwargs=title_kwargs,
        hist_kwargs=hist_kwargs,
    )
    fig.savefig(f'{save_dir}/corner_outcome.png', dpi=300)
    plt.close(fig)
    logger.info("VIS: `corner_for_outcome()` finished.")


def plot_velocity_spin_mass(mc_result, outcome_filter=None, save_dir=OUTPUT_DIR_IMG):
    """
    Scatter plot: velocity magnitude vs spin, colored by mass.

    Parameters
    ----------
    mc_result : MonteCarloResult
        Output from monte_carlo_19D.
    outcome_filter : str or list[str], optional
        Filter for specific outcome types, e.g., "Creative_ionized".
        If None, all stars are plotted.
    """
    data = mc_result.all_star_outcomes

    if data.size == 0:
        print("No data to plot.")
        return

    # Apply outcome filter if requested
    if outcome_filter is not None:
        if isinstance(outcome_filter, str):
            outcome_filter = [outcome_filter]
        mask = np.isin(data['outcome'], outcome_filter)
        data = data[mask]

    # Extract columns
    v = data['v_mag']  # km/s
    spin = data['spin']  # 1/s
    mass = data['mass_Msun']  # Msun

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(v, spin, c=mass, cmap='viridis', s=10, edgecolor='none', alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"Mass $\left(M_{\odot}\right)$", fontsize=14)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Velocity magnitude $\left(\text{km}~\text{s}^{-1}\right)$", fontsize=14)
    ax.set_ylabel(r"Spin $\left(\text{s}^{-1}\right)$", fontsize=14)

    ax.text(
        0.95,
        0.95,
        f"Total Plotted = {len(v)}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round",  # or "square", "round,pad=0.2", etc.
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
            linewidth=1.0,
        ),
    )

    fig.tight_layout()
    fig.savefig(f'{save_dir}/velocity_spin_mass.png', dpi=300)
    plt.close(fig)
    logger.info("VIS: `plot_velocity_spin_mass()` finished.")
