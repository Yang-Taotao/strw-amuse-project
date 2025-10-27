"""
Stellar Cluster Simulation with Intruder Star and GIF output with intruder highlighted
Updated intruder placement closer to cluster edge,
and finer timestep & longer simulation for smoother GIF.
"""

# import

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from amuse.support import options
from amuse.lab import Particles, constants
from amuse.units import units, nbody_system
from amuse.ic.plummer import new_plummer_sphere
from amuse.ext.salpeter import new_salpeter_mass_distribution
from amuse.community.ph4.interface import ph4

# config
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"
options.GlobalOptions.instance().override_value_for_option(
    name="polling_internval_in_milliseconds", value=10
)

# alias
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "media"))
os.makedirs(MEDIA_DIR, exist_ok=True)


# func


def create_stellar_cluster(N_stars=100, cluster_mass=1000, cluster_radius=1):
    converter = nbody_system.nbody_to_si(
        cluster_mass | units.MSun, cluster_radius | units.parsec
    )
    cluster = new_plummer_sphere(N_stars, converter)
    masses = new_salpeter_mass_distribution(
        N_stars,
        mass_min=0.1 | units.MSun,
        mass_max=50.0 | units.MSun,
    )
    scale = (cluster_mass | units.MSun) / masses.sum()
    cluster.mass = masses * scale
    cluster.radius = 0.01 | units.parsec
    print(
        f"Cluster mass {cluster.mass.sum().in_(units.MSun)}, "
        f"size {cluster.position.lengths().max().in_(units.parsec)}"
    )
    return cluster, converter


def setup_gravity(converter):
    gravity = ph4(converter)
    gravity.parameters.timestep_parameter = 0.001
    gravity.parameters.epsilon_squared = (0.05 | units.parsec) ** 2
    return gravity


def create_intruder_star(cluster_com, intruder_mass=10):
    # Fixed position: 6 parsec along x-axis from cluster center
    approach_distance = 6.0 | units.parsec
    center_vals = cluster_com.value_in(units.parsec)
    offset = np.array([approach_distance.value_in(units.parsec), 0.0, 0.0])
    pos_vals = center_vals + offset
    positions = pos_vals.reshape((1, 3)) | units.parsec

    # Keep the initial velocity calculation as before:
    vvir = (constants.G * (1000 | units.MSun) / approach_distance) ** 0.5
    vel_vals = np.array([-0.5 * vvir.value_in(units.kms), 0.0, 0.0])
    velocities = vel_vals.reshape((1, 3)) | units.kms

    intruder = Particles(1)
    intruder.mass = intruder_mass | units.MSun
    intruder.position = positions
    intruder.velocity = velocities
    intruder.radius = 0.01 | units.parsec
    print(
        f"Intruder placed at {intruder.position.in_(units.parsec)}, "
        f"velocity = {intruder.velocity}"
    )
    return intruder


def plot_cluster_evolution(times, energies, history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(times.value_in(units.Myr), energies["total"].value_in(units.J))
    ax1.set_title("Total Energy Evolution")
    ax1.set_xlabel("Time (Myr)")
    ax1.set_ylabel("Energy (J)")
    ax1.grid(True)
    virial = -2 * energies["kinetic"] / energies["potential"]
    ax2.plot(times.value_in(units.Myr), virial)
    ax2.axhline(1.0, color="r", linestyle="--", label="Equilibrium")
    ax2.set_title("Virial Ratio Evolution")
    ax2.set_xlabel("Time (Myr)")
    ax2.set_ylabel("-2T/U")
    ax2.legend()
    ax2.grid(True)
    for ti in [0, len(history) // 2, -1]:
        pos = history[ti]
        tval = times[ti].value_in(units.Myr)
        ax3.scatter(pos[:, 0], pos[:, 1], s=1, label=f"t={tval:.2f} Myr")
    ax3.set_title("Cluster XY Snapshots")
    ax3.set_xlabel("X (pc)")
    ax3.set_ylabel("Y (pc)")
    ax3.legend()
    ax3.grid(True)
    ax3.axis("equal")
    final = history[-1]
    ax4.scatter(final[:, 0], final[:, 1], c=final[:, 2], s=2, cmap="viridis")
    ax4.set_title("Final State (colored by Z)")
    ax4.set_xlabel("X (pc)")
    ax4.set_ylabel("Y (pc)")
    ax4.grid(True)
    ax4.axis("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MEDIA_DIR, "cluster_evolution.png"), dpi=150, bbox_inches="tight"
    )


def create_evolution_gif_plane(history, times, plane="xy", interval=0.2, filename=None):

    if filename is None:
        filename = os.path.join(MEDIA_DIR, f"cluster_evolution_{plane}.gif")

    selected_indices = []
    last_time = -interval
    for i, t in enumerate(times):
        t_myr = t.value_in(units.Myr)
        if t_myr - last_time >= interval or i == 0:
            selected_indices.append(i)
            last_time = t_myr

    fig, ax = plt.subplots(figsize=(6, 6))

    # Set axis labels and limits
    if plane == "xy":
        ax.set_xlabel("X (pc)")
        ax.set_ylabel("Y (pc)")
        x_idx, y_idx = 0, 1
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
    elif plane == "yz":
        ax.set_xlabel("Y (pc)")
        ax.set_ylabel("Z (pc)")
        x_idx, y_idx = 1, 2
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
    elif plane == "xz":
        ax.set_xlabel("X (pc)")
        ax.set_ylabel("Z (pc)")
        x_idx, y_idx = 0, 2
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
    else:
        raise ValueError(f"Unknown plane: {plane}")

    cluster_scat = ax.scatter([], [], s=5, color="blue", label="Cluster Stars")
    intruder_scat = ax.scatter(
        [], [], s=80, color="red", marker="*", label="Intruder Star"
    )

    def update(frame):
        pos = history[selected_indices[frame]]
        cluster_scat.set_offsets(pos[:-1, [x_idx, y_idx]])
        intruder_scat.set_offsets(pos[-1:, [x_idx, y_idx]])
        ax.set_title(
            f"Cluster Evolution ({plane.upper()} plane): "
            f"t = {times[selected_indices[frame]].value_in(units.Myr):.2f} Myr"
        )
        ax.legend(loc="upper right")
        return cluster_scat, intruder_scat

    ani = animation.FuncAnimation(
        fig, update, frames=len(selected_indices), interval=10, repeat_delay=1000
    )
    ani.save(filename, writer="pillow")
    plt.close(fig)
    print(f"GIF saved to {filename}")


def main():
    N_stars = 42
    cluster_mass = 100  # MSun
    cluster_radius = 1  # parsec
    t_end = 3.0 | units.Myr
    dt = 0.01 | units.Myr

    cluster, converter = create_stellar_cluster(N_stars, cluster_mass, cluster_radius)
    gravity = setup_gravity(converter)
    gravity.particles.add_particles(cluster)

    times = []
    energies = {"kinetic": [], "potential": [], "total": []}
    history = []

    t = 0.0 | units.Myr
    half = t_end / 2
    while t < half:
        t += dt
        gravity.evolve_model(t)
        times.append(t)
        history.append(gravity.particles.position.value_in(units.parsec).copy())
        ke = gravity.kinetic_energy
        pe = gravity.potential_energy
        energies["kinetic"].append(ke)
        energies["potential"].append(pe)
        energies["total"].append(ke + pe)
        print(
            f"t={t.value_in(units.Myr):.2f} Myr, "
            f"E_total={(ke+pe).value_in(units.J):.2e} J, "
            f"Virial={(-2*ke/pe):.3f}"
        )

    com = gravity.particles.center_of_mass()
    dists = (gravity.particles.position - com).lengths()
    size = dists.max()
    # Place intruder 4 parsec beyond cluster edge (fix: explicit units addition)
    intruder_distance = size + (4.0 | units.parsec)
    intruder = create_intruder_star(com, intruder_mass=10)
    gravity.particles.add_particles(intruder)

    while t < t_end:
        t += dt
        gravity.evolve_model(t)
        times.append(t)
        history.append(gravity.particles.position.value_in(units.parsec).copy())
        ke = gravity.kinetic_energy
        pe = gravity.potential_energy
        energies["kinetic"].append(ke)
        energies["potential"].append(pe)
        energies["total"].append(ke + pe)
        dist = (gravity.particles[-1].position - com).length()
        print(
            f"t={t.value_in(units.Myr):.2f} Myr, "
            f"E_total={(ke+pe).value_in(units.J):.2e} J, "
            f"Virial={(-2*ke/pe):.3f}, "
            f"Intruder dist={dist.value_in(units.parsec):.2f} pc"
        )

    times_arr = np.array([tt.value_in(units.Myr) for tt in times]) | units.Myr
    for key in energies:
        energies[key] = np.array([e.value_in(units.J) for e in energies[key]]) | units.J

    plot_cluster_evolution(times_arr, energies, history)
    create_evolution_gif_plane(history, times, plane="xy", interval=0.01)
    create_evolution_gif_plane(history, times, plane="yz", interval=0.01)
    create_evolution_gif_plane(history, times, plane="xz", interval=0.01)
    gravity.stop()
    print(
        "Simulation completed successfully. Output: cluster_evolution.png and cluster_evolution.gif"
    )


if __name__ == "__main__":
    np.random.seed(42)
    main()
