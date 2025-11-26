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

from helpers import (
    make_seba_stars,
    make_triple_binary_system,
    make_sph_from_two_stars,
    detect_close_pair,
    run_fi_collision,
    critical_velocity,
    outcomes,
    transformation_to_cartesian,
    # compute_remnant_spin,
)

from config import (
    OUTPUT_DIR_COLLISIONS,
    OUTPUT_DIR_FINAL_STATES,
    OUTPUT_DIR_LOGS,
    OUTPUT_DIR_SNAPSHOTS,
)


def run_6_body_simulation(
    sep,
    true_anomalies,
    ecc,
    theta,
    phi,
    v_mag,
    impact_parameter,
    psi,
    distance,
    run_label,
    masses=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    centers=None,  # <-- impact orientation angles
    age=3.5,
):
    """
    Run a full 6-body simulation combining stellar dynamics, stellar evolution, and hydrodynamic mergers.

    Uses spherical coordinates for incoming binaries. The first binary is fixed at the origin
    with orbit in the xy-plane and direction 0. True anomalies (phases) and impact orientations (psi)
    can be specified for all three binaries.
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

    # Set units
    target_age = age | units.Myr
    t_end = 100 | units.yr
    dt = 0.1 | units.yr
    t = 0 | units.yr

    # Local init
    frames = []
    n_collision = 0

    centers, v_vectors, directions, orbit_plane, phases = transformation_to_cartesian(
        sep=sep,
        true_anomalies=true_anomalies,
        ecc=ecc,
        theta=theta,
        phi=phi,
        v_mag=v_mag,
        distance=distance,
    )

    # Default psi if not provided
    if psi is None:
        psi = [0.0, 0.0, 0.0]

    # --------------------------
    # Stellar evolution setup
    # --------------------------
    seba, seba_particles = make_seba_stars(masses, target_age)

    grav_particles = make_triple_binary_system(
        masses=masses,
        seps=sep,
        ecc=ecc,
        directions=directions,
        orbit_plane=orbit_plane[1:],  # only incoming binaries B and C
        impact_parameter=impact_parameter,
        centers=centers,
        v_coms=v_vectors,
        phases=phases,
        psi=psi,
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

        # Save pre-collision snapshot
        pre_snapshot = gravity.particles.copy()
        frames.append(pre_snapshot)

        # Collision detection
        radii = [p.radius for p in gravity.particles]
        pair = detect_close_pair(gravity.particles, radii)

        if pair:
            i, j, sep_distance = pair
            key_i = gravity.particles[i].key
            key_j = gravity.particles[j].key

            print(
                f"Collision detected at {t.value_in(units.yr):.1f} yr between keys {key_i} and {key_j}"
            )

            success, remnant = collision(
                key_i, key_j, n_collision, gravity, seba, key_map, t, run_label
            )

            if success:
                n_collision += 1

                # Save post-collision snapshot (remnant included)
                post_snapshot = gravity.particles.copy()
                frames.append(post_snapshot)

                if remnant is None:
                    # Destructive collision → no remnant
                    print(
                        f"Destructive collision: simulation stopped at t = {t.value_in(units.yr):.1f} yr."
                    )
                    break

                # Skip to next timestep after collision
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
            outcome = outcomes(initial_particles, final_particles)
            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(
                final_particles, final_filename, "amuse", overwrite_file=True
            )

            return frames, outcome
        else:
            max_mass_particle = max(final_particles, key=lambda p: p.mass)
            max_mass = max_mass_particle.mass
            max_velocity = max_mass_particle.velocity.length()

            print(
                f"Most massive star: Mass = {max_mass.value_in(units.MSun):.2f} MSun, "
                f"Velocity = {max_velocity.value_in(units.kms):.2f} km/s"
            )

            # Determine final outcome

            outcome = outcomes(initial_particles, final_particles)
            print("Final outcome of the system:", outcome)

            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(
                final_particles, final_filename, "amuse", overwrite_file=True
            )

            return frames, outcome


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


def collision(key_i, key_j, n_collision, gravity, seba, key_map, t, run_label=""):
    """
    Handle a single stellar collision using Fi SPH and produce a bound remnant.
    The remnant **replaces one of the colliding particles** in gravity.particles
    to preserve pre/post collision continuity.
    """
    try:
        # --- fetch particles by key ---
        p_i = next((p for p in gravity.particles if p.key == key_i), None)
        p_j = next((p for p in gravity.particles if p.key == key_j), None)

        if p_i is None or p_j is None:
            print(f"⚠️ Collision aborted: keys {key_i},{key_j} not found.")
            return False, None

        # Build SPH initial conditions
        colliders = Particles()
        colliders.add_particle(p_i.copy())
        colliders.add_particle(p_j.copy())
        sph = make_sph_from_two_stars(colliders, n_sph_per_star=500)
        if len(sph) == 0:
            print("⚠️ SPH particle set empty — destructive collision")
            return remove_colliders(gravity, seba, key_map, [key_i, key_j])

        # --- Pre-collision COM and velocity ---
        pre_com_pos = (p_i.mass * p_i.position + p_j.mass * p_j.position) / (
            p_i.mass + p_j.mass
        )
        pre_com_vel = (p_i.mass * p_i.velocity + p_j.mass * p_j.velocity) / (
            p_i.mass + p_j.mass
        )

        # Center SPH
        sph.position -= sph.center_of_mass()
        sph.velocity -= sph.center_of_mass_velocity()

        # Save SPH initial state
        final_filename = os.path.join(
            OUTPUT_DIR_COLLISIONS,
            f"collision_{n_collision}_sph_input_{run_label}.amuse",
        )
        write_set_to_file(sph, final_filename, "amuse", overwrite_file=True)

        # Run Fi
        gas_out, diag = run_fi_collision(sph, t_end=0.1 | units.yr)
        print("Fi collision done:", diag)

        # --- Bound particle selection ---
        r = gas_out.position - gas_out.center_of_mass()
        v = gas_out.velocity - gas_out.center_of_mass_velocity()
        r_mag = r.lengths()
        m_total = gas_out.total_mass()
        phi_pot = -(constants.G * m_total) / (r_mag + (1 | units.RSun))
        e_spec = 0.5 * v.lengths() ** 2 + phi_pot
        bound_mask = e_spec.value_in(units.m**2 / units.s**2) < 0.0

        if not np.any(bound_mask):
            print("No bound particles found: destructive collision")
            return remove_colliders(gravity, seba, key_map, [key_i, key_j])

        bound_particles = gas_out[bound_mask]
        m_bound = bound_particles.total_mass()
        remnant_radius = (m_bound.value_in(units.MSun) ** 0.57) | units.RSun

        # --- Very small remnant → destructive ---
        if m_bound <= (5 | units.MSun):
            print("⚠️ Very small remnant mass; destructive collision")
            return remove_colliders(gravity, seba, key_map, [key_i, key_j])

        # --- Compute remnant velocity preserving SPH internal motion ---
        sph_com_vel_before_shift = gas_out.center_of_mass_velocity()
        remnant_vel = pre_com_vel + (
            bound_particles.center_of_mass_velocity() - sph_com_vel_before_shift
        )

        # --- Replace p_i with remnant properties ---
        # Compute remnant properties
        remnant_mass = m_bound
        remnant_radius = remnant_radius
        remnant_pos = pre_com_pos
        remnant_vel = pre_com_vel + (
            bound_particles.center_of_mass_velocity()
            - gas_out.center_of_mass_velocity()
        )

        # Overwrite p_i with remnant
        p_i.mass = remnant_mass
        p_i.radius = remnant_radius
        p_i.position = remnant_pos
        p_i.velocity = remnant_vel

        # Remove p_j only
        gravity.particles.remove_particle(p_j)
        gravity.recommit_particles()

        # Update SEBA
        remnant_seba = Particle()
        remnant_seba.mass = remnant_mass
        remnant_seba.radius = remnant_radius
        remnant_seba.age = (3.5 | units.Myr) + t
        seba.particles.add_particle(remnant_seba)
        seba.recommit_particles()

        # Update key_map
        key_map[p_i.key] = remnant_seba

        print(
            f"Collision {n_collision} processed: remnant = {m_bound.value_in(units.MSun):.2f} M☉, "
            f"R = {remnant_radius.value_in(units.RSun):.2f} R☉ (replacing {key_i}, removed {key_j})"
        )

        return True, p_i

    except Exception as e:
        print("⚠️ Collision handling failed with exception type:", type(e))
        print("Exception details:", e)
        return False, None


def remove_colliders(gravity, seba, key_map, keys):
    """
    Remove colliders from gravity and SEBA by particle keys.
    Only removes particles if they exist.
    Returns: (True, removed_keys)
    """
    removed_keys = []

    # --- Gravity ---
    to_remove_grav = Particles()
    for k in keys:
        p = next((p for p in gravity.particles if p.key == k), None)
        if p is not None:
            to_remove_grav.add_particle(p)
            removed_keys.append(k)
            key_map.pop(k, None)
    if len(to_remove_grav) > 0:
        gravity.particles.remove_particles(to_remove_grav)
        gravity.recommit_particles()

    # --- SEBA ---
    to_remove_seba = Particles()
    for k in keys:
        s = key_map.get(k, None)
        if s is not None and s in seba.particles:
            to_remove_seba.add_particle(s)
            key_map.pop(k, None)
        else:
            # fallback
            s = next((s for s in seba.particles if getattr(s, "key", None) == k), None)
            if s is not None:
                to_remove_seba.add_particle(s)
                key_map.pop(k, None)
    if len(to_remove_seba) > 0:
        seba.particles.remove_particles(to_remove_seba)
        seba.recommit_particles()

    return True, removed_keys
