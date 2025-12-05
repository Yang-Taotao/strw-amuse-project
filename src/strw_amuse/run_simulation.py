"""
Main simulation functions for 6-body encounter simulation.
Combining stellar dynamics, stellar evolution, and hydrodynamic mergers.
"""

# imports
import os
import time

from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file  # , read_set_from_file
from amuse.units import nbody_system, units

from src.strw_amuse.collision import collision
from src.strw_amuse.config import (
    OUTPUT_DIR_COLLISIONS,
    OUTPUT_DIR_COLLISIONS_DIAGNOSTICS,
    OUTPUT_DIR_FINAL_STATES,
    OUTPUT_DIR_LOGS,
    OUTPUT_DIR_OUTCOMES,
    OUTPUT_DIR_SNAPSHOTS,
)
from src.strw_amuse.helpers import (
    make_seba_stars,
    make_triple_binary_system,
    outcomes,
    transformation_to_cartesian,
)
from src.strw_amuse.logging_config import setup_logging

# from amuse.lab import *


def run_6_body_simulation(
    sep,
    ecc,
    v_mag,
    impact_parameter,
    theta,
    phi,
    psi,
    distance,
    true_anomalies,
    run_label,
    masses=None,
    centers=None,  # <-- impact orientation angles
    age=3.5,
):
    """
    Run a full 6-body simulation
    combining stellar dynamics, stellar evolution, and hydrodynamic mergers.

    Uses spherical coordinates for incoming binaries.
    The first binary is fixed at the origin with orbit in the xy-plane and direction 0.
    True anomalies (phases) and impact orientations (psi) can be specified for all three binaries.
    """

    # Create directories
    output_dirs = (
        OUTPUT_DIR_COLLISIONS,
        OUTPUT_DIR_FINAL_STATES,
        OUTPUT_DIR_LOGS,
        OUTPUT_DIR_SNAPSHOTS,
        OUTPUT_DIR_COLLISIONS_DIAGNOSTICS,
        OUTPUT_DIR_OUTCOMES,
    )
    for d in output_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # initialize logging (MPI-aware)
    logger = setup_logging(log_dir=OUTPUT_DIR_LOGS)
    logger.info("Logging initialized for simulation: %s", run_label)

    # Set units
    target_age = age | units.Myr
    t_end = 100 | units.yr
    dt = 0.1 | units.yr
    t = 0 | units.yr

    # Local init
    frames = []
    n_collision = 0

    # default masses (avoid mutable default argument)
    if masses is None:
        masses = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

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

    logger.info("Starting simulation")
    start = time.time()
    max_time = 20 * 60
    gravity.stopping_conditions.collision_detection.enable()

    collision_history = []
    # Main evolution loop
    while t < t_end:
        if time.time() - start > max_time:
            logger.warning(
                "Runtime > %.1f min -> End sim at t=%.1f yr.", max_time / 60, t.value_in(units.yr)
            )
            break
        t += dt
        gravity.evolve_model(t)
        seba.evolve_model(target_age + t)

        # Save pre-collision snapshot
        pre_snapshot = gravity.particles.copy()
        frames.append(pre_snapshot)

        # Collision detection
        # collision_pairs = gravity.collision_detection()

        if gravity.stopping_conditions.collision_detection.is_set():
            sc = gravity.stopping_conditions.collision_detection
            p1 = sc.particles(0)[0]
            p2 = sc.particles(1)[0]
            key_i, key_j = p1.key, p2.key

            logger.info(
                "Collision detected at %.1f yr between keys %s, %s",
                t.value_in(units.yr),
                key_i,
                key_j,
            )

            success, remnant = collision(
                key_i, key_j, n_collision, gravity, seba, key_map, t, run_label
            )
            if success:
                n_collision += 1
                frames.append(gravity.particles.copy())

                collision_history.append([key_i, key_j])

                if remnant is None:
                    logger.warning("Destructive collision -> stopping simulation")
                    break

                # Skip to next timestep after collision
                continue

    # BEFORE stopping gravity
    final_particles = gravity.particles.copy()

    gravity.stop()
    seba.stop()

    if len(final_particles) == 0:
        logger.warning("No particles remaining in the system! Returning defaults.")
        max_mass_particle = None
        max_mass = 0 | units.MSun
        max_velocity = 0 | units.kms
        # Save final system
        final_filename = os.path.join(OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse")
        write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)

        # Normalize return shape: always return (frames, outcome)
        return frames, None
    else:
        if t < t_end:
            outcome = outcomes(
                initial_particles,
                final_particles,
                collision_history,
                run_label=run_label,
            )
            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)

            return frames, outcome
        else:
            max_mass_particle = max(final_particles, key=lambda p: p.mass)
            max_mass = max_mass_particle.mass
            max_velocity = max_mass_particle.velocity.length()

            # Determine final outcome

            outcome = outcomes(
                initial_particles,
                final_particles,
                collision_history,
                run_label=run_label,
            )
            logger.info("Final outcome of the system: %s", outcome)

            # Save final system
            final_filename = os.path.join(
                OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
            )
            write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)

            return frames, outcome
