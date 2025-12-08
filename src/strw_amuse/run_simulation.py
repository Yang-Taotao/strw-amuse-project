"""
Main simulation functions for 6-body encounter simulation.
Combining stellar dynamics, stellar evolution, and hydrodynamic mergers.
"""

import logging
import os
import time

from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file
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
    transformation_to_cartesian,
)
from src.strw_amuse.stopping import (
    find_bound_groups,
    group_com,
    group_physical_size,
    is_ionized_single,
    outcomes,
    specific_pair_energy,
)

logger = logging.getLogger(__name__)


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
    centers=None,
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
    # Default psi if not provided
    if psi is None:
        psi = [0.0, 0.0, 0.0]

    centers, v_vectors, directions, orbit_plane, phases = transformation_to_cartesian(
        sep=sep,
        true_anomalies=true_anomalies,
        ecc=ecc,
        theta=theta,
        phi=phi,
        v_mag=v_mag,
        distance=distance,
    )

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
    check_every = 10 | units.yr
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

        # Periodic check every 10 yrs
        # -> it stops simulation when either:
        # 1. desired outcome has happened or 2. has not happened and not about to happen.
        if t >= check_every:
            check_every += 10 | units.yr

            particles = gravity.particles
            n_part = len(particles)

            # 1) Desired outcome: any single (ionized) star > MASS_THRESHOLD
            MASS_THRESHOLD = 70.0 | units.MSun
            massive_indices = [i for i, p in enumerate(particles) if p.mass > MASS_THRESHOLD]

            for idx in massive_indices:
                if is_ionized_single(idx, particles):
                    mass_msun = particles[idx].mass.in_(units.MSun).number  # extract float
                    logger.info(
                        "Desired outcome seen at t=%.1f yr: particle %s mass=%.3f Msun is ionized.",
                        t.value_in(units.yr),
                        particles[idx].key,
                        mass_msun,
                    )

                    # finalize and exit returning outcome
                    final_particles = gravity.particles.copy()
                    gravity.stop()
                    seba.stop()
                    outcome = outcomes(
                        initial_particles, final_particles, collision_history, run_label=run_label
                    )
                    final_filename = os.path.join(
                        OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
                    )
                    write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)
                    return frames, outcome

            # 2) Check if system is dilute & unbound:
            # min pairwise distance > FAR_DISTANCE and no bound pairs
            FAR_DISTANCE = 200.0 | units.AU
            min_pair_distance = None
            any_bound_pair = False
            for i in range(n_part):
                for j in range(i + 1, n_part):
                    d = (particles[i].position - particles[j].position).length()
                    if (min_pair_distance is None) or (d < min_pair_distance):
                        min_pair_distance = d
                    E = specific_pair_energy(particles[i], particles[j])
                    # specific_pair_energy returns specific energy (m^2 / s^2) — test sign directly
                    if E.value_in(units.m**2 / units.s**2) < 0:
                        any_bound_pair = True

            if (
                (min_pair_distance is not None)
                and (min_pair_distance > FAR_DISTANCE)
                and (not any_bound_pair)
            ):
                logger.info(
                    "System is dilute & unbound at t=%.1f yr (min distance=%.1f AU). Stopping.",
                    t.value_in(units.yr),
                    min_pair_distance.in_(units.AU),
                )
                final_particles = gravity.particles.copy()
                gravity.stop()
                seba.stop()
                outcome = outcomes(
                    initial_particles, final_particles, collision_history, run_label=run_label
                )
                final_filename = os.path.join(
                    OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
                )
                write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)
                return frames, outcome

            # 3) If there are bound groups,
            # check if they are compact and mutually well-separated (stable)
            groups = find_bound_groups(particles)
            if len(groups) >= 1:
                group_sizes = [group_physical_size(particles, g) for g in groups]
                group_coms = [group_com(particles, g) for g in groups]

                compact_threshold = 100.0 | units.AU
                all_compact = all(
                    (sz < compact_threshold) or (len(g) == 1) for sz, g in zip(group_sizes, groups)
                )

                well_separated = True
                STABLE_FACTOR = 50.0  # increase factor to make criterion looser
                for i_g in range(len(groups)):
                    for j_g in range(i_g + 1, len(groups)):
                        # ignore singletons when considering group separation
                        if len(groups[i_g]) == 1 and len(groups[j_g]) == 1:
                            continue
                        sep = (group_coms[i_g] - group_coms[j_g]).length()
                        size_max = max(group_sizes[i_g], group_sizes[j_g])
                        if size_max.value_in(units.AU) == 0:
                            size_max = 1.0 | units.AU

                        if sep <= (STABLE_FACTOR * size_max):
                            well_separated = False
                            break
                    if not well_separated:
                        break

                if all_compact and well_separated:
                    logger.info(
                        "System consists of compact bound groups mutually well-separated at "
                        "t=%.1f yr -> declaring stable and stopping.",
                        t.value_in(units.yr),
                    )
                    final_particles = gravity.particles.copy()
                    gravity.stop()
                    seba.stop()
                    outcome = outcomes(
                        initial_particles, final_particles, collision_history, run_label=run_label
                    )
                    final_filename = os.path.join(
                        OUTPUT_DIR_FINAL_STATES, f"final_system_{run_label}.amuse"
                    )
                    write_set_to_file(final_particles, final_filename, "amuse", overwrite_file=True)
                    return frames, outcome
