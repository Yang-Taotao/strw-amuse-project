

#import librarie


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from amuse.units import units


def visualize_frames(frames, run_label="test"):
    """
    Visualize AMUSE simulation frames and produce a GIF.

    Centered on the most massive star in the final frame.
    Tracks all other stars relative to that one.
    Color-coded by stellar mass (with colorbar).
    """

    if len(frames) == 0:
        print("⚠️ No frames provided.")
        return

    os.makedirs("Gif-6body", exist_ok=True)

    # --- Gather all masses for color normalization ---
    all_masses = np.array([p.mass.value_in(units.MSun) for f in frames for p in f])
    m_min, m_max = all_masses.min(), all_masses.max()
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=m_min, vmax=m_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    def mass_to_color(mass):
        return cmap(norm(mass))

    # --- Find most massive star in final frame ---
    final_frame = frames[-1]
    final_masses = np.array([p.mass.value_in(units.MSun) for p in final_frame])
    max_index = np.argmax(final_masses)
    print(f"Tracking most massive star (index {max_index}) with final mass {final_masses[max_index]:.2f} M☉")

    # Extract its position at each frame (to recenter)
    tracked_positions = np.array([
        [
            f[max_index].x.value_in(units.AU),
            f[max_index].y.value_in(units.AU)
        ]
        for f in frames
    ])

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter([], [], s=[])
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', color='black')

    ax.set_xlabel("x [AU]", fontsize=12)
    ax.set_ylabel("y [AU]", fontsize=12)
    ax.set_title("Centered on most massive star", fontsize=13)

    # Colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mass [M$_\\odot$]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # --- Initialization ---
    def init():
        sc.set_offsets(np.empty((0, 2)))
        ax.set_xlim(-1200, 1200)
        ax.set_ylim(-1200, 1200)
        return sc, time_text

    # --- Frame update ---
    def update(frame_index):
        frame = frames[frame_index]
        x = np.array([p.x.value_in(units.AU) for p in frame])
        y = np.array([p.y.value_in(units.AU) for p in frame])
        masses = np.array([p.mass.value_in(units.MSun) for p in frame])

        sizes = np.clip(masses * 2, 10, 500)
        colors = [mass_to_color(m) for m in masses]

        # Center all coordinates on tracked star
        x_rel = x - tracked_positions[frame_index, 0]
        y_rel = y - tracked_positions[frame_index, 1]

        sc.set_offsets(np.c_[x_rel, y_rel])
        sc.set_sizes(sizes)
        sc.set_color(colors)

        ax.set_xlim(-1200, 1200)
        ax.set_ylim(-1200, 1200)

        dt = 5  # years per frame
        t = frame_index * dt
        time_text.set_text(f"t = {t:.0f} yr")

        return sc, time_text

    # --- Animation ---
    ani = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, interval=50, blit=False, repeat=False)

    gif_filename = os.path.join("Gif-6body", f"encounter_evolution_{run_label}.gif")
    writer = PillowWriter(fps=12)
    ani.save(gif_filename, writer=writer)

    print(f"🎬 GIF saved as {gif_filename}")
    plt.close(fig)
