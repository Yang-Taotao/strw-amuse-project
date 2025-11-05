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

from src.strw_amuse.config import OUTPUT_DIR_GIF

# func repo


def visualize_frames(frames, run_label="test") -> None:
    """
    Visualize AMUSE simulation frames and produce a GIF.

    Centered on the most massive star in the final frame.
    Tracks all other stars relative to that one.
    Color-coded by stellar mass (with colorbar).
    """

    # Create output directory if none exists
    os.makedirs(name=OUTPUT_DIR_GIF, exist_ok=True)

    # Exception for empty frames
    if len(frames) == 0:
        print("Caution: No frames provided. Exitting visualization.")
        return None

    # Gather all masses for color normalization
    all_masses = np.array(
        object=[p.mass.value_in(units.MSun) for f in frames for p in f]
    )
    m_min, m_max = all_masses.min(), all_masses.max()

    # Color map setup
    cmap = plt.get_cmap(name="plasma")
    norm = mcolors.Normalize(vmin=m_min, vmax=m_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_to_color(mass):
        return cmap(norm(mass))

    # Find most massive star in final frame
    final_frame = frames[-1]
    final_masses = np.array(object=[p.mass.value_in(units.MSun) for p in final_frame])
    max_idx = np.argmax(a=final_masses)
    print(
        f"""Tracking:
        Most massive star at idx {max_idx} with final mass {final_masses[max_idx]:.2f} M$_\\odot$
        """
    )

    # Extract its position at each frame (to recenter)
    tracked_positions = np.array(
        object=[
            [f[max_idx].x.value_in(units.AU), f[max_idx].y.value_in(units.AU)]
            for f in frames
        ]
    )

    # Figure setup
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter([], [], s=[])
    time_text = ax.text(
        x=0.02,
        y=0.95,
        s="",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="black",
    )

    ax.set_xlabel(xlabel="x [AU]", fontsize=12)
    ax.set_ylabel(ylabel="y [AU]", fontsize=12)
    ax.set_title(label="Centered on most massive star", fontsize=13)

    # Colorbar
    cbar = plt.colorbar(mappable=sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label="Mass [M$_\\odot$]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Initialization
    def init():
        sc.set_offsets(offsets=np.empty((0, 2)))
        ax.set_xlim(left=-1200, right=1200)
        ax.set_ylim(bottom=-1200, top=1200)
        return sc, time_text

    # Frame update
    def update(frame_idx):
        frame = frames[frame_idx]
        x = np.array([p.x.value_in(units.AU) for p in frame])
        y = np.array([p.y.value_in(units.AU) for p in frame])
        masses = np.array([p.mass.value_in(units.MSun) for p in frame])

        sizes = np.clip(masses * 2, 10, 500)
        colors = [mass_to_color(m) for m in masses]

        # Center all coordinates on tracked star
        x_rel = x - tracked_positions[frame_idx, 0]
        y_rel = y - tracked_positions[frame_idx, 1]

        sc.set_offsets(np.c_[x_rel, y_rel])
        sc.set_sizes(sizes)
        sc.set_color(colors)

        ax.set_xlim(-1200, 1200)
        ax.set_ylim(-1200, 1200)

        dt = 5  # years per frame
        t = frame_idx * dt
        time_text.set_text(f"t = {t:.0f} yr")

        return sc, time_text

    # Animation
    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(frames),
        init_func=init,
        interval=50,
        blit=False,
        repeat=False,
    )

    gif_filename = os.path.join(OUTPUT_DIR_GIF, f"encounter_evolution_{run_label}.gif")
    writer = PillowWriter(fps=12)
    ani.save(filename=gif_filename, writer=writer)

    print(f"GIF saved at {gif_filename}")
    plt.close(fig=fig)
