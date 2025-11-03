"""
6-body encounter simulation combining stellar dynamics, stellar evolution, and hydrodynamic mergers.
"""

# import
import os
import numpy as np

from amuse.lab import *
from amuse.units import units
from amuse.io import write_set_to_file, read_set_from_file
from amuse.datamodel import Particles

from amuse.community.ph4.interface import ph4

from src.strw_amuse.helpers import (
    make_seba_stars,
    make_triple_binary_system,
    make_sph_from_two_stars,
    detect_close_pair,
    run_fi_collision,
    # compute_remnant_spin,
)


from src.strw_amuse.config import (
    OUTPUT_DIR_COLLISIONS,
    OUTPUT_DIR_FINAL_STATES,
    OUTPUT_DIR_LOGS,
    OUTPUT_DIR_SNAPSHOTS,
)

# func repo


def run_6_body_simulation(
    age, masses, sep, ecc, direction, centers, v_coms, run_label=""
):
    """
    Run a full 6-body simulation
    Combining stellar dynamics, stellar evolution, and hydrodynamic mergers.

    1. Set up three binaries, evolves them under gravity, and applies stellar evolution through SEBA
    2. Auto detect and resolve physical collisions between stars
    3. When a collision is detected, invokes an Smoothed Particle Hydrodynamics calculation with Fi\
        to simulate the merger and generate a realistic remnant with updated mass, radius,\
        and velocity.
    4. Collisions, pre/post states, and SPH outputs are saved to disk

    Parameters
    ----------
    age : float
        Initial stellar age in Myr, used for SEBA stellar evolution initialization.
    masses : list of float
        Masses (in MSun) of the six stars (two per binary).
    sep : list of float
        Initial semi-major axes (in AU) for each of the three binaries.
    ecc : list of float
        Orbital eccentricities of the binaries.
    direction : list of float
        Common orientation vector for all binaries (defines orbital plane orientation).
    centers : list of 3-element lists
        Positions (in AU) of each binary's center of mass in the simulation frame.
    v_coms : list of 3-element lists
        Center-of-mass velocities (in km/s) for each binary.
    run_label : str, optional
        Label used to name output files for this specific simulation run.

    Returns
    -------
    frames : list of Particles sets
        Snapshots of the system after each collision or major event, for visualization or replay.
    max_mass : ScalarQuantity
        Mass of the most massive star remaining at the end of the simulation (in MSun).
    max_velocity : ScalarQuantity
        Velocity magnitude of that most massive star (in km/s).

    Notes
    -----
    - Collisions are detected dynamically by comparing interstellar distances to stellar radii with 
        a configurable buffer factor.
    - SPH mergers are performed using Fi with automatic scaling of mass and size units to maintain
        numerical stability.
    - Each collision produces pre- and post-collision snapshots, and merged remnants are reinserted
        into the N-body system with their new properties.
    - If a collision fails or Fi crashes, the merger is skipped, and the system continues with
        default remnant parameters.
    - The simulation terminates if all stars are ejected or merged into a single object.
    """

    # Create directories
    output_dirs = (
        OUTPUT_DIR_COLLISIONS,
        OUTPUT_DIR_FINAL_STATES,
        OUTPUT_DIR_LOGS,
        OUTPUT_DIR_SNAPSHOTS,
    )
    for d in output_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    target_age = age | units.Myr
    t_end = 2000 | units.yr
    dt = 5 | units.yr
    t = 0 | units.yr

    frames = []
    n_collision = 0
    last_collision_pair = None

    # Stellar evolution setup
    seba, seba_particles = make_seba_stars(masses, target_age)
    grav_particles = make_triple_binary_system(
        masses, sep, ecc, direction, centers, v_coms
    )

    total_mass = grav_particles.total_mass()
    length_scale = 1000 | units.AU
    converter = nbody_system.nbody_to_si(total_mass, length_scale)
    gravity = ph4(converter)
    gravity.particles.add_particles(grav_particles)

    for g, s in zip(gravity.particles, seba.particles):
        g.mass = s.mass
        g.radius = s.radius

    print("Starting simulation")

    # Main evolution loop
    while t < t_end:
        t += dt
        gravity.evolve_model(t)
        seba.evolve_model(target_age + t)
        frames.append(gravity.particles.copy())

        radii = [p.radius for p in gravity.particles]
        # --- Collision detection ---
        pair = detect_close_pair(gravity.particles, radii)

        if pair:
            i, j, sep = pair
            print(
                f"Collision detected at {t.value_in(units.yr):.1f} yr between {i} and {j}"
            )

            success, remnant = collision(i, j, n_collision, gravity, seba, t, run_label)

            if success:
                post_snapshot = gravity.particles.copy()
                frames.append(post_snapshot)
                n_collision += 1

    # BEFORE stopping gravity
    final_particles = gravity.particles.copy()

    gravity.stop()
    seba.stop()

    if len(final_particles) == 0:
        print(" No particles remaining in the system! Returning defaults.")
        max_mass_particle = None
        max_mass = 0 | units.MSun
        max_velocity = 0 | units.kms
    else:
        max_mass_particle = max(final_particles, key=lambda p: p.mass)
        max_mass = max_mass_particle.mass
        max_velocity = max_mass_particle.velocity.length()

        print(
            f"Most massive star: Mass = {max_mass.value_in(units.MSun):.2f} MSun, "
            f"Velocity = {max_velocity.value_in(units.kms):.2f} km/s"
        )

    # Save final system
    final_filename = os.path.join(
        OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
    )
    write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)

    return frames, max_mass, max_velocity


def collision(i, j, n_collision, gravity, seba, t, run_label=""):
    """
    Handle a single stellar collision event using Fi SPH and physically consistent remnant construction.
    """

    try:
        # --- Save pre-collision snapshot ---
        pre_filename = os.path.join(
            OUTPUT_DIR_COLLISIONS,
            f"pre_collision_{n_collision}_{run_label}.amuse",
        )
        write_set_to_file(
            gravity.particles.copy(), pre_filename, "amuse", overwrite_file=True
        )

        # --- Extract colliding stars ---
        p_i = gravity.particles[i]
        p_j = gravity.particles[j]

        # --- Build SPH initial conditions ---
        colliders = Particles()
        colliders.add_particle(p_i.copy())
        colliders.add_particle(p_j.copy())
        sph = make_sph_from_two_stars(
            colliders, n_sph_per_star=500
        )  # adjust for performance

        if len(sph) == 0:
            print("Caution: SPH particle set empty — skipping.")
            return False, None

        # Center on COM before Fi run
        com_pos = sph.center_of_mass()
        com_vel = sph.center_of_mass_velocity()
        sph.position -= com_pos
        sph.velocity -= com_vel

        # Save SPH input
        write_set_to_file(
            sph,
            os.path.join(
                OUTPUT_DIR_COLLISIONS,
                f"collision_{n_collision}_sph_input_{run_label}.amuse",
            ),
            "amuse",
            overwrite_file=True,
        )

        # --- Run Fi hydrodynamics ---
        gas_out, diag = run_fi_collision(sph, t_end=0.1 | units.yr)
        print("Fi collision done:", diag)
        # map Fi output back to global coordinates
        local_com_pos = gas_out.center_of_mass()
        local_com_vel = gas_out.center_of_mass_velocity()
        # compute global pair COM using progenitors (mass-weighted)
        global_pair_com_pos = (p_i.mass * p_i.position + p_j.mass * p_j.position) / (
            p_i.mass + p_j.mass
        )
        global_pair_com_vel = (p_i.mass * p_i.velocity + p_j.mass * p_j.velocity) / (
            p_i.mass + p_j.mass
        )

        gas_out.position += global_pair_com_pos - local_com_pos
        gas_out.velocity += global_pair_com_vel - local_com_vel

        # Save SPH output
        write_set_to_file(
            gas_out,
            os.path.join(
                OUTPUT_DIR_COLLISIONS,
                f"collision_{n_collision}_sph_output_{run_label}.amuse",
            ),
            "amuse",
            overwrite_file=True,
        )

        # --- Compute per-particle energies (approximate binding check) ---
        com_pos = gas_out.center_of_mass()
        com_vel = gas_out.center_of_mass_velocity()
        r = gas_out.position - com_pos
        v = gas_out.velocity - com_vel

        phi = -(constants.G * gas_out.total_mass()) / (r.lengths() + (1 | units.RSun))
        E_kin = 0.5 * v.lengths() ** 2
        E_spec = E_kin + phi  # specific total energy (m²/s²)
        bound_mask = E_spec < 0 | (units.m**2 / units.s**2)

        # ensure boolean array
        bound_mask = np.array(bound_mask)
        bound_indices = np.where(bound_mask)[0]
        bound_particles = (
            gas_out[bound_indices] if len(bound_indices) > 0 else gas_out.copy()
        )

        if len(bound_particles) == 0:
            print("Caution: No bound particles found; fallback to full SPH set.")

        # --- Compute remnant properties ---
        Mbound = bound_particles.total_mass()
        COM_pos = bound_particles.center_of_mass()
        COM_vel = bound_particles.center_of_mass_velocity()

        # Mass–radius relation
        remnant_radius = (Mbound.value_in(units.MSun) ** 0.57) | units.RSun

        # Momentum conservation using progenitor momenta
        total_momentum = p_i.mass * p_i.velocity + p_j.mass * p_j.velocity
        new_velocity = total_momentum / Mbound

        # --- Replace star i with remnant, remove j ---
        p_i.mass = Mbound
        p_i.position = COM_pos
        p_i.velocity = new_velocity
        p_i.radius = remnant_radius

        gravity.particles.remove_particle(p_j)
        gravity.recommit_particles()

        print(
            f"Collision {n_collision} processed: remnant = "
            f"{Mbound.value_in(units.MSun):.2f} M☉, R = {remnant_radius.value_in(units.RSun):.2f} R☉"
        )

        return True, p_i

    except Exception as e:
        print(f"Collision handling failed: {e}")
        return False, None
