"""
Main simulation functions for 6-body encounter simulation.
Combining stellar dynamics, stellar evolution, and hydrodynamic mergers.
"""

# imports
import os
import numpy as np

# from amuse.lab import *

from amuse.units import units, nbody_system, constants
from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file  # , read_set_from_file
from amuse.datamodel import Particle, Particles

from src.strw_amuse.helpers import (
    make_seba_stars,
    make_triple_binary_system,
    make_sph_from_two_stars,
    detect_close_pair,
    run_fi_collision,
    critical_velocity,
    outcomes,
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
    sep,
    ecc,
    direction,
    v_coms,
    orbit_plane,
    impact_parameter,
    distance,
    masses=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    centers=None,
    age=3.5,
    run_label="",
):
    """
    Run a full 6-body simulation combining stellar dynamics, stellar evolution, and hydrodynamic mergers.

    This function sets up three binaries, evolves them under gravity, applies stellar evolution through SEBA,
    and automatically detects and resolves physical collisions between stars.When a collision is detected,
    the function invokes an SPH (Smoothed Particle Hydrodynamics) calculation using the Fi code to simulate
    the merger and generate a realistic remnant with updated mass, radius, and velocity. Collisions, pre/post states,
    and SPH outputs are saved to disk for later analysis.

    Parameters
    ----------
    sep : list of float
        Initial semi-major axes (in AU) for each of the three binaries.
        3 seperations for 3 binaries
    ecc : list of float
        Orbital eccentricities of the binaries.
        3 eccentricities for 3 binaries
    direction : list of float
        orbital orientation for 2 incoming binaries. The other binary resides at rest at the origin.
        default = 0 for the first binary, the other two are given by this input
    centers : list of 3-element lists
        Positions (in AU) of 2 binary's center of mass in the simulation frame. The other binary resides at the origin.
        default = [0,0,0] for the first binary, the other two are given by this input
    v_coms : list of 3-element lists
        Center-of-mass velocities with respect to the escape velocity(in km/s) for 2 binaries. The other binary resides at rest at the origin.
        default = [0,0,0] for the first binary, the other two are given by this input
    orbit_plane : list of 3-element lists
        Normal vectors defining the orbital planes for the two incoming binaries. The other binary resides in the xy-plane.
        default = [0,0,1] for the first binary, the other two are given by this input
    impact_parameter : list of float
        Impact parameters (in AU) for the two incoming binaries relative to the target binary at the origin.
        default = 0 for the first binary, the other two are given by this input
    masses : list of float
        Masses (in MSun) of the six stars (two per binary). If not provided, defaults to equal 50 MSun stars.
    age : float
        Initial age of the stars in Myr for stellar evolution setup. Defaults to 3.5 Myr.
    run_label : str, optional
        Label used to name output files for this specific simulation run.
    Total of free parameters: 12 ( 3 sep, 3 ecc, 2 directions, 2 centers, 2 v_coms)

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
    - Collisions are detected dynamically by comparing interstellar distances to stellar radii with a
      configurable buffer factor.
    - SPH mergers are performed using Fi with automatic scaling of mass and size units to maintain
      numerical stability.
    - Each collision produces pre- and post-collision snapshots, and merged remnants are reinserted
      into the N-body system with their new properties.
    - If a collision fails or Fi crashes, the merger is skipped, and the system continues with
      default remnant parameters.
    - The simulation terminates if all stars are ejected or merged into a single object.
    """

    # create directories
    output_dirs = (
        OUTPUT_DIR_COLLISIONS,
        OUTPUT_DIR_FINAL_STATES,
        OUTPUT_DIR_LOGS,
        OUTPUT_DIR_SNAPSHOTS,
    )
    for d in output_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # set units
    target_age = age | units.Myr
    t_end = 100 | units.yr
    dt = 0.1 | units.yr
    t = 0 | units.yr

    # local init
    frames = []
    n_collision = 0
    # last_collision_pair = None

    # default distance for centers
    if centers is None:
        centers = [
            [0, 0, 0],
            [distance[0], distance[0], distance[0]],
            [distance[1], distance[1], distance[1]],
        ]  # in AU

    # shift centers based on impact parameters
    v_crit = critical_velocity(masses, sep, ecc)
    # Express input dimensionless velocities in physical units <<- !ATTENTION: NEED VECTORIZATION!
    v_coms[1] = [v_crit * v for v in v_coms[1]]
    v_coms[2] = [v_crit * v for v in v_coms[2]]

    # Stellar evolution setup
    seba, seba_particles = make_seba_stars(masses, target_age)
    grav_particles = make_triple_binary_system(
        masses, sep, ecc, direction, orbit_plane, impact_parameter, centers, v_coms
    )
    initial_particles = grav_particles.copy()
    total_mass = grav_particles.total_mass()
    length_scale = 1000 | units.AU
    converter = nbody_system.nbody_to_si(total_mass, length_scale)
    gravity = ph4(converter)
    gravity.particles.add_particles(grav_particles)

    key_map = {g.key: s for g, s in zip(gravity.particles, seba.particles)}

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

            success, remnant = collision(
                i, j, n_collision, gravity, seba, t, key_map, run_label
            )

            if success:
                # Save post-collision snapshot
                post_snapshot = gravity.particles.copy()
                frames.append(post_snapshot)
                n_collision += 1

                # Check if a remnant exists
                if remnant is None:
                    print(
                        f"Destructive collision: sim ended at t = {t.value_in(units.yr):.1f} yr."
                    )

                    break
                continue

    # BEFORE stopping gravity
    final_particles = gravity.particles.copy()

    gravity.stop()
    seba.stop()

    if len(final_particles) == 0:
        print(" No particles remaining in the system! Returning defaults.")
        max_mass_particle = None
        max_mass = 0 | units.MSun
        max_velocity = 0 | units.kms
        # Save final system
        final_filename = os.path.join(
            OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
        )
        write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)

        return frames
    else:
        if t < t_end:
            outcome = outcomes(initial_particles, final_particles, max_mass=None)
            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(
                final_particles, final_filename, "amuse", overwrite_file=True
            )

            return frames
        else:
            max_mass_particle = max(final_particles, key=lambda p: p.mass)
            max_mass = max_mass_particle.mass
            max_velocity = max_mass_particle.velocity.length()

            print(
                f"Most massive star: Mass = {max_mass.value_in(units.MSun):.2f} MSun, "
                f"Velocity = {max_velocity.value_in(units.kms):.2f} km/s"
            )

            # Determine final outcome

            outcome = outcomes(initial_particles, final_particles, max_mass)
            print("Final outcome of the system:", outcome)

            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(
                final_particles, final_filename, "amuse", overwrite_file=True
            )

            return frames


# --- 2) Helper to find a seba particle by id ---
def find_seba_by_id(seba_set, pid):
    matches = [p for p in seba_set if getattr(p, "id", None) == pid]  # <<- ATTENTION
    return matches[0] if matches else None


def find_seba_by_key(seba_set, key):
    """Return the SEBA particle with the same AMUSE key, if it exists."""
    for p in seba_set:
        if p.key == key:
            return p
    return None


def collision(i, j, n_collision, gravity, seba, t, key_map, run_label=""):
    """
    Handle a single stellar collision using Fi SPH and produce a bound remnant.
    Safely removes original colliders from gravity and SEBA,
    handles destructive collisions (very small remnants), and reinserts the remnant if appropriate.
    """
    try:
        pre_particles = gravity.particles.copy()
        # print("Pre-collision particles (key,mass):",
        #      [(p.key, p.mass.value_in(units.MSun)) for p in pre_particles])

        # Get colliding stars
        p_i = gravity.particles[i]
        p_j = gravity.particles[j]
        # print(f"Collision between gravity indices {i},{j} -> keys {p_i.key},{p_j.key}; "
        #      f"masses {p_i.mass.value_in(units.MSun):.2f}, {p_j.mass.value_in(units.MSun):.2f}")

        # Build SPH initial conditions
        colliders = Particles()
        colliders.add_particle(p_i.copy())
        colliders.add_particle(p_j.copy())
        sph = make_sph_from_two_stars(colliders, n_sph_per_star=500)

        # If SPH fails or empty, treat as destructive collision
        if len(sph) == 0:
            print("⚠️ SPH particle set empty — treating as destructive collision.")
            return remove_colliders(gravity, seba, key_map, p_i, p_j, destructive=True)

        # Center SPH particles <<- ATTENTION
        sph.position -= sph.center_of_mass()
        sph.velocity -= sph.center_of_mass_velocity()

        # Save SPH initial state
        final_filename = os.path.join(
            OUTPUT_DIR_COLLISIONS,
            f"collision_{n_collision}_sph_input_{run_label}.amuse",
        )
        write_set_to_file(
            sph,
            final_filename,
            "amuse",
            overwrite_file=True,
        )

        # Run Fi
        gas_out, diag = run_fi_collision(sph, t_end=0.1 | units.yr)
        print("Fi collision done:", diag)

        # Map SPH back to progenitor COM
        progenitor_com_pos = (p_i.mass * p_i.position + p_j.mass * p_j.position) / (
            p_i.mass + p_j.mass
        )
        progenitor_com_vel = (p_i.mass * p_i.velocity + p_j.mass * p_j.velocity) / (
            p_i.mass + p_j.mass
        )
        gas_out.position += progenitor_com_pos - gas_out.center_of_mass()
        gas_out.velocity += progenitor_com_vel - gas_out.center_of_mass_velocity()

        # --- Bound particle selection ---
        com_pos = gas_out.center_of_mass()
        com_vel = gas_out.center_of_mass_velocity()
        r = gas_out.position - com_pos
        v = gas_out.velocity - com_vel
        r_mag = r.lengths()
        m_total = gas_out.total_mass()
        phi = -(constants.G * m_total) / (r_mag + (1 | units.RSun))
        e_spec = 0.5 * v.lengths() ** 2 + phi
        bound_mask = e_spec.value_in(units.m**2 / units.s**2) < 0.0

        if np.any(bound_mask):
            bound_particles = gas_out[bound_mask]
            m_bound = bound_particles.total_mass()
            com_pos = bound_particles.center_of_mass()
            com_vel = bound_particles.center_of_mass_velocity()
            remnant_radius = (m_bound.value_in(units.MSun) ** 0.57) | units.RSun
        else:
            print("No bound particles found: destructive collision")
            return remove_colliders(gravity, seba, key_map, p_i, p_j, destructive=True)

        # --- Destructive collision check ---
        if m_bound <= (5 | units.MSun):
            print("⚠️ Very small remnant mass; treating as destructive collision")
            remove_colliders(gravity, seba, key_map, p_i, p_j, destructive=True)
            return True, None

        # --- Normal remnant creation ---
        remnant = Particle()
        remnant.mass = m_bound
        remnant.radius = remnant_radius
        remnant.position = com_pos
        remnant.velocity = com_vel

        # Remove originals and add remnant
        remove_colliders(gravity, seba, key_map, p_i, p_j, destructive=False)
        gravity.particles.add_particle(remnant)
        gravity.recommit_particles()

        # Add remnant to SEBA
        remnant_seba = Particle()
        remnant_seba.mass = remnant.mass
        remnant_seba.radius = remnant.radius
        remnant_seba.age = (3.5 | units.Myr) + t
        seba.particles.add_particle(remnant_seba)
        seba.recommit_particles()

        key_map[remnant.key] = remnant_seba

        print(
            f"Collision {n_collision} processed: remnant = "
            f"{m_bound.value_in(units.MSun):.2f} M☉, R = {remnant_radius.value_in(units.RSun):.2f} R☉"
        )
        return True, remnant

    except Exception as e:
        print("⚠️ Collision handling failed with exception type:", type(e))
        print("Exception details:", e)
        return False, None


def remove_colliders(gravity, seba, key_map, p_i, p_j, destructive=True):
    """
    Safely remove colliding stars from gravity and SEBA.
    If destructive is True, do not insert any remnant.
    """
    # Gravity removal
    to_remove_grav = Particles()
    for p in (p_i, p_j):
        if p in gravity.particles:
            to_remove_grav.add_particle(p)
            key_map.pop(p.key, None)
    if len(to_remove_grav) > 0:
        gravity.particles.remove_particles(to_remove_grav)
        gravity.recommit_particles()

    # SEBA removal
    to_remove_seba = Particles()
    for p in (p_i, p_j):
        # Look up SEBA particle by key instead of object identity
        s_obj = next((s for s in seba.particles if s.key == p.key), None)
        if s_obj is not None:
            to_remove_seba.add_particle(s_obj)
            key_map.pop(p.key, None)
    if len(to_remove_seba) > 0:
        seba.particles.remove_particles(to_remove_seba)
        seba.recommit_particles()

    if destructive:
        return True, None
    return True
