"""
Plotting utilities for AMUSE simulation.
"""

# import
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from amuse.units import units

from src.strw_amuse.config import OUTPUT_DIR_GIF, OUTPUT_DIR_IMG


# func repo
def visualize_frames(frames, run_label="test", massive_threshold=70.0):
    """
    Visualize AMUSE simulation frames and produce a GIF.

    For the final frame, finds all stars with mass >= massive_threshold M☉.
    Produces one subplot per massive star.
    Each subplot tracks the corresponding star through all frames.
    Axis limits are fixed to [-100, 100] AU.
    Color-coded by stellar mass with a shared colorbar.
    """

    if len(frames) == 0:
        print("Warning: No frames provided.")
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
    print(f"Tracking {n_massive} massive stars in final frame")

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
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=all_masses.min(), vmax=all_masses.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_to_color(m):
        return cmap(norm(m))

    # --- Figure and subplots setup ---
    fig, axes = plt.subplots(1, n_massive, figsize=(6 * n_massive, 6))
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
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

    # --- Add a shared colorbar ---
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label("Mass [M$_\odot$]", fontsize=12)
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
    print(f"GIF saved as {gif_filename}")


def visualize_initial_final_frames(frames, run_label="test"):
    """
    Clean visualization of initial vs final simulation frames.
    Includes:
    - Initial binary orbital arcs (first 40 frames)
    - Final-frame trajectories of all surviving stars over all time
    - Robust handling of mergers (missing particles)
    """
    # ---------------------------------------------------------
    # 0. Safety checks
    # ---------------------------------------------------------
    if len(frames) == 0:
        print("Warning: No frames provided.")
        return

    output_dir = OUTPUT_DIR_IMG
    os.makedirs(output_dir, exist_ok=True)

    initial = frames[0]
    final = frames[-1]

    # ---------------------------------------------------------
    # 1. Build colormap for masses
    # ---------------------------------------------------------
    all_masses = np.array([p.mass.value_in(units.MSun) for f in frames for p in f])
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=all_masses.min(), vmax=all_masses.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_color(m):
        return cmap(norm(m))

    # ---------------------------------------------------------
    # 2. Track most massive star for final centering
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
    # 3. Prepare figure
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

    for ax in (ax1, ax2):
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")

    ax1.set_title("Initial Configuration")
    ax2.set_title("Final Configuration")
    plt.colorbar(sm, ax=[ax1, ax2], label="Mass [M$_\\odot$]")

    # ---------------------------------------------------------
    # 4. Helper to scatter stars
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
    # 5. Initial frame: center on COM
    # ---------------------------------------------------------
    mi = np.array([p.mass.value_in(units.MSun) for p in initial])
    xi = np.array([p.x.value_in(units.AU) for p in initial])
    yi = np.array([p.y.value_in(units.AU) for p in initial])

    com_x = np.sum(mi * xi) / np.sum(mi)
    com_y = np.sum(mi * yi) / np.sum(mi)

    xi_rel, yi_rel = scatter_stars(ax1, initial, center=(com_x, com_y))

    # ---------------------------------------------------------
    # 6. Plot initial velocities
    # ---------------------------------------------------------
    if hasattr(initial[0], "vx"):
        vx = np.array([p.vx.value_in(units.AU / units.yr) for p in initial])
        vy = np.array([p.vy.value_in(units.AU / units.yr) for p in initial])
        ax1.quiver(
            xi_rel,
            yi_rel,
            vx,
            vy,
            angles="xy",
            scale_units="xy",
            scale=1.5,
            color="gray",
            alpha=0.6,
            width=0.004,
        )

    # ---------------------------------------------------------
    # 7. Initial internal binary orbital arcs (first 40 frames)
    # ---------------------------------------------------------
    binary_groups = [(0, 1), (2, 3), (4, 5)]
    orbit_colors = ["red", "blue", "green"]

    n_traj = min(40, len(frames))

    for bidx, (i, j) in enumerate(binary_groups):

        # Skip missing binaries (due to mergers)
        if i >= len(initial) or j >= len(initial):
            continue

        # initial static positions (for anchoring arcs)
        xi0 = initial[i].x.value_in(units.AU)
        yi0 = initial[i].y.value_in(units.AU)
        xj0 = initial[j].x.value_in(units.AU)
        yj0 = initial[j].y.value_in(units.AU)

        traj_i = []
        traj_j = []

        for k in range(n_traj):
            fr = frames[k]
            if i >= len(fr) or j >= len(fr):
                break

            xi_k = fr[i].x.value_in(units.AU)
            yi_k = fr[i].y.value_in(units.AU)
            xj_k = fr[j].x.value_in(units.AU)
            yj_k = fr[j].y.value_in(units.AU)

            mi_k = fr[i].mass.value_in(units.MSun)
            mj_k = fr[j].mass.value_in(units.MSun)

            # binary COM
            bcx = (mi_k * xi_k + mj_k * xj_k) / (mi_k + mj_k)
            bcy = (mi_k * yi_k + mj_k * yj_k) / (mi_k + mj_k)

            # relative orbital motion + anchor to initial COM-center frame
            traj_i.append([xi_k - bcx + (xi0 - com_x), yi_k - bcy + (yi0 - com_y)])

            traj_j.append([xj_k - bcx + (xj0 - com_x), yj_k - bcy + (yj0 - com_y)])

        traj_i = np.array(traj_i)
        traj_j = np.array(traj_j)

        if len(traj_i) > 1:
            ax1.plot(traj_i[:, 0], traj_i[:, 1], color=orbit_colors[bidx], alpha=0.7)
        if len(traj_j) > 1:
            ax1.plot(traj_j[:, 0], traj_j[:, 1], color=orbit_colors[bidx], alpha=0.7)

    # ---------------------------------------------------------
    # 8. Final frame scatter centered on tracked star
    # ---------------------------------------------------------
    x2_rel, y2_rel = scatter_stars(ax2, final, center=(cx_final, cy_final))

    # ---------------------------------------------------------
    # 9. Final velocities
    # ---------------------------------------------------------
    if hasattr(final[0], "vx"):
        vx = np.array([p.vx.value_in(units.AU / units.yr) for p in final])
        vy = np.array([p.vy.value_in(units.AU / units.yr) for p in final])
        ax2.quiver(
            x2_rel,
            y2_rel,
            vx * 40,
            vy * 40,
            angles="xy",
            scale_units="xy",
            scale=1.4,
            color="gray",
            alpha=0.6,
            width=0.004,
        )

    # ---------------------------------------------------------
    # 10. Full trajectories for surviving stars (centered on tracked star)
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
            ax2.plot(traj_x, traj_y, alpha=0.6)

    # Adaptive limits
    dx = max(50, (x2_rel.max() - x2_rel.min()) * 1.3)
    dy = max(50, (y2_rel.max() - y2_rel.min()) * 1.3)
    ax2.set_xlim(-dx, dx)
    ax2.set_ylim(-dy, dy)

    # ---------------------------------------------------------
    # 11. Final layout
    # ---------------------------------------------------------
    plt.suptitle(f"Initial and Final Frames: {run_label}", fontsize=16)

    png_filename = os.path.join(OUTPUT_DIR_IMG, f"encounter_evolution_{run_label}.png")
    plt.savefig(png_filename)
    print(f"Comparison png saved as {png_filename}")
    plt.close(fig)

    return None
