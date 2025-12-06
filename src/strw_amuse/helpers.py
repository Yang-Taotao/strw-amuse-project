"""
General helper functions for AMUSE simulation.
"""

import os

import numpy as np
from amuse.community.seba.interface import SeBa
from amuse.datamodel import Particle, Particles
from amuse.datamodel.particle_attributes import bound_subset
from amuse.io import write_set_to_file
from amuse.units import constants, units
from amuse.units.quantities import VectorQuantity

from src.strw_amuse.config import OUTPUT_DIR_COLLISIONS_OUTCOMES


def make_triple_binary_system(
    masses,
    seps,
    ecc,
    directions,
    orbit_plane,
    impact_parameter,
    centers=None,
    v_coms=None,
    phases=None,  # true anomalies
    psi=None,  # impact orientation for incoming binaries
):
    """
    Create a system of three interacting binaries with fully tunable parameters.
    """

    if not (len(masses) == 6 and len(seps) == 3 and len(directions) == 3):
        raise ValueError("Expect masses=6, seps=3, directions=3.")

    # Default centers
    if centers is None:
        centers = [[-300, 0, 0], [300, 0, 0], [0, 600, 0]]
    if v_coms is None:
        v_coms = [[0, 0, 0], [10, 0, 0], [-10, 0, 0]]  # first binary at rest
    if phases is None:
        phases = [0.0, 0.0, 0.0]
    if psi is None:
        psi = [0.0, 0.0, 0.0]

    ma1, ma2, mb1, mb2, mc1, mc2 = masses
    sep_a, sep_b, sep_c = seps
    dir_a, dir_b, dir_c = directions
    ecc_a, ecc_b, ecc_c = ecc
    f_a, f_b, f_c = phases
    psi_b, psi_c = psi

    # Convert centers and velocities to VectorQuantity
    centers_q = [VectorQuantity(c, units.AU) for c in centers]
    v_coms_q = [VectorQuantity(v, units.kms) for v in v_coms]

    # Create binaries
    p1, p2 = make_binary(
        ma1,
        ma2,
        sep_a | units.AU,
        ecc_a,
        center=centers_q[0],
        direction=dir_a,
        orbit_plane=[0, 0, 1],
        impact_parameter=0.0 | units.AU,
        f=f_a,
    )
    p3, p4 = make_binary(
        mb1,
        mb2,
        sep_b | units.AU,
        ecc_b,
        center=centers_q[1],
        direction=dir_b,
        orbit_plane=orbit_plane[0],
        impact_parameter=impact_parameter[0] | units.AU,
        f=f_b,
        psi=psi_b,
    )
    p5, p6 = make_binary(
        mc1,
        mc2,
        sep_c | units.AU,
        ecc_c,
        center=centers_q[2],
        direction=dir_c,
        orbit_plane=orbit_plane[1],
        impact_parameter=impact_parameter[1] | units.AU,
        f=f_c,
        psi=psi_c,
    )

    # Name particles
    p1.name, p2.name = "A1", "A2"
    p3.name, p4.name = "B1", "B2"
    p5.name, p6.name = "C1", "C2"

    # Apply center-of-mass velocities
    for p in (p1, p2):
        p.velocity += v_coms_q[0]
    for p in (p3, p4):
        p.velocity += v_coms_q[1]
    for p in (p5, p6):
        p.velocity += v_coms_q[2]

    # Combine all particles
    particles = Particles()
    for p in [p1, p2, p3, p4, p5, p6]:
        particles.add_particle(p)

    return particles


def make_binary(
    m1,
    m2,
    a,
    e=0.0,
    center=None,
    direction=0.0,
    orbit_plane=None,
    impact_parameter=0.0 | units.AU,
    f=0.0,  # true anomaly
    psi=0.0,  # impact orientation
):
    """
    Create a binary system with arbitrary eccentricity, orbit plane, phase, and impact orientation.
    """

    # avoid mutable default for orbit_plane
    if orbit_plane is None:
        orbit_plane = [0, 0, 1]

    plane = np.array(orbit_plane, dtype=float)
    plane /= np.linalg.norm(plane)

    # Reference vector perpendicular to orbit plane
    ref = np.array([1, 0, 0]) if abs(np.dot(plane, [1, 0, 0])) < 0.9 else np.array([0, 1, 0])
    off_set_dir = np.cross(plane, ref)
    off_set_dir /= np.linalg.norm(off_set_dir)

    # Rotate offset by direction and psi
    c2, s2 = np.cos(direction + psi), np.sin(direction + psi)
    r_dir = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    off_set_dir = np.dot(r_dir, off_set_dir)

    # Masses
    m1 = m1 | units.MSun
    m2 = m2 | units.MSun
    total_mass = m1 + m2

    # Center including impact parameter along offset
    if not isinstance(center, VectorQuantity):
        center = VectorQuantity(center, units.AU)
    center += off_set_dir * impact_parameter

    # Positions using true anomaly f
    r_rel = a * (1 - e**2) / (1 + e * np.cos(f))
    r1 = -(m2 / total_mass) * r_rel
    r2 = (m1 / total_mass) * r_rel

    # Orbital velocities
    if e == 0.0:
        v_rel = (constants.G * total_mass / a) ** 0.5
    elif e < 1.0:
        v_rel = (constants.G * total_mass * (1 + e) / (a * (1 - e))) ** 0.5
    else:
        raise ValueError("Eccentricity must be < 1")

    v1 = +(m2 / total_mass) * v_rel
    v2 = -(m1 / total_mass) * v_rel

    # Base positions and velocities in XY plane with phase rotation
    pos1_base = np.array(
        [r1.value_in(units.AU) * np.cos(f), r1.value_in(units.AU) * np.sin(f), 0.0]
    )
    pos2_base = np.array(
        [r2.value_in(units.AU) * np.cos(f), r2.value_in(units.AU) * np.sin(f), 0.0]
    )
    vel1_base = np.array(
        [-v1.value_in(units.kms) * np.sin(f), v1.value_in(units.kms) * np.cos(f), 0.0]
    )
    vel2_base = np.array(
        [-v2.value_in(units.kms) * np.sin(f), v2.value_in(units.kms) * np.cos(f), 0.0]
    )

    # Rotate to orbit plane
    n = plane
    z_axis = np.array([0.0, 0.0, 1.0])
    v_cross = np.cross(z_axis, n)
    s = np.linalg.norm(v_cross)
    c = np.dot(z_axis, n)
    if s == 0:
        r_plane = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array(
            [
                [0, -v_cross[2], v_cross[1]],
                [v_cross[2], 0, -v_cross[0]],
                [-v_cross[1], v_cross[0], 0],
            ]
        )
        r_plane = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / s**2)

    c2, s2 = np.cos(direction), np.sin(direction)
    r_dir = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    r_total = np.dot(r_plane, r_dir)

    pos1 = np.dot(r_total, pos1_base) | units.AU
    pos2 = np.dot(r_total, pos2_base) | units.AU
    vel1 = np.dot(r_total, vel1_base) | units.kms
    vel2 = np.dot(r_total, vel2_base) | units.kms

    p1 = Particle(mass=m1)
    p2 = Particle(mass=m2)
    p1.position = center + pos1
    p2.position = center + pos2
    p1.velocity = vel1
    p2.velocity = vel2

    return p1, p2


def make_seba_stars(masses_msun, age):
    """
    masses_msun: list of floats (Msun)
    age: quantity with units (e.g. 3.5 | units.Myr)
    returns: seba, seba.particles (Particles with .mass, .radius, etc.)
    """
    seba = SeBa()  # fast SSE-style stellar evolution
    stars = Particles()
    for m in masses_msun:
        p = Particle(mass=m | units.MSun)
        stars.add_particle(p)
    seba.particles.add_particles(stars)
    seba.evolve_model(age)
    return seba, seba.particles


def critical_velocity(masses, sep, ecc):
    """
    Calculate the critical velocity for a binary-binary-binary encounter.
    Parameters:
    -------------
    masses : list of float
        Masses (in MSun) of the six stars (two per binary).
    sep : list of float
        Orbital separations (in AU) of the three binaries.
    ecc : list of float
        Orbital eccentricities of the three binaries.
    Returns:
    -------------
    v_crit : float
        Critical velocity (in km/s) for the encounter.
    """
    G = constants.G.in_(units.AU**3 / (units.MSun * units.day**2))  # AU^3 / (MSun * day^2)
    G = G.value_in(units.kms**2 * units.AU / units.MSun)  # km^2/s^2 * AU / MSun

    m1, m2, m3, m4, m5, m6 = masses
    a1, a2, a3 = sep
    e1, e2, e3 = ecc

    mu1 = (m1 * m2) / (m1 + m2)
    mu2 = (m3 * m4) / (m3 + m4)
    mu3 = (m5 * m6) / (m5 + m6)
    # === <<- section in between can be vectorized

    energy_1 = -G * m1 * m2 / (2 * a1 * (1 - e1**2))
    energy_2 = -G * m3 * m4 / (2 * a2 * (1 - e2**2))
    energy_3 = -G * m5 * m6 / (2 * a3 * (1 - e3**2))

    total_energy = energy_1 + energy_2 + energy_3
    reduced_mass = (mu1 * mu2) / (mu1 + mu2 + mu3)

    v_crit = np.sqrt(-2 * total_energy / reduced_mass)  # in km/s

    return v_crit


def outcomes(
    initial_particles,
    final_particles,
    collision_history,
    massive_threshold=70 | units.MSun,
    creative_threshold=10 | units.MSun,
    run_label="sim",
    output_dir=OUTPUT_DIR_COLLISIONS_OUTCOMES,
):
    """
    Compute and save outcomes for each final star using explicit collision mapping.

    Stores all outcomes in an AMUSE file, returns the 'most interesting' outcome
    (stars with exactly 1 collision) for quick printing.

    Parameters:
    -----------
    initial_particles : Particles
        Initial stars.
    final_particles : Particles
        Final stars.
    collision_history : list of [key_i, key_j]
        Keys of stars that collided.
    massive_threshold : quantity
        Mass above which a remnant is considered massive.
    creative_threshold : quantity
        Mass below which a collision is destructive.
    run_label : str
        Label used for output filename.
    output_dir : str
        Directory to save outcomes AMUSE file.

    Returns:
    --------
    most_interesting : list of dict
        Stars that participated in exactly 1 collision.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(final_particles, list):
        final_particles = Particles(final_particles)
        # <<-ERROR
        # type list[Unkonwn] cannot be assigned to param size of type int in function __init__

    # Build mapping: final star key -> number of collisions
    collisions_per_star = {p.key: 0 for p in final_particles}
    for key_i, key_j in collision_history:
        for star in final_particles:
            if star.key == key_i or star.key == key_j:
                collisions_per_star[star.key] += 1
                break

    # Create Particles set for AMUSE output
    outcome_particles = Particles()

    summary_outcome = []

    for star in final_particles:
        ncoll = collisions_per_star.get(star.key, 0)
        M = star.mass

        # Determine outcome
        if ncoll == 0:
            outcome_type = "No_collision"
            ncomp = 0
        elif M < creative_threshold:
            outcome_type = "Destructive"
            ncomp = 0
        elif M < massive_threshold:
            outcome_type = "Creative_not_massive"
            ncomp = 0
        else:
            # Massive star: bound vs ionized
            bound = bound_subset(final_particles, core=star)
            ncomp = len(bound) - 1
            outcome_type = "Creative_bound" if ncomp > 0 else "Creative_ionized"

        # Add a particle for this star with attributes storing outcome info
        p = Particle()
        p.key = star.key
        p.mass = M
        p.radius = star.radius
        p.position = star.position
        p.velocity = star.velocity
        # custom attributes for outcome info
        p.outcome_type = outcome_type
        p.n_collisions = ncoll
        p.n_companions = ncomp
        outcome_particles.add_particle(p)

        # Collect most interesting outcomes for print
        if ncoll >= 1:
            summary_outcome.append(
                {
                    "star_key": star.key,
                    "outcome": outcome_type,
                    "collisions": ncoll,
                    "mass_Msun": M.value_in(units.MSun),
                    "n_companions": ncomp,
                }
            )

    # Save outcomes to AMUSE file
    filename = os.path.join(output_dir, f"outcomes_{run_label}.amuse")
    write_set_to_file(outcome_particles, filename, "amuse", overwrite_file=True)

    return summary_outcome


def transformation_to_cartesian(
    sep,
    true_anomalies,
    ecc,
    theta,
    phi,
    v_mag,
    distance,
):
    """
    Compute initial conditions for a 6-body triple binary system with
    incoming binaries B and C approaching from opposite directions,
    using independent theta and phi for each binary.

    Parameters:
        theta : list of 2 floats, polar angles for B and C
        phi   : list of 2 floats, azimuthal angles for B and C
        v_mag : list of 2 floats, velocities in units of fraction of critical

    Returns:
        centers, v_vectors, directions, orbit_plane, phases
    """
    if len(theta) != 2 or len(phi) != 2 or len(v_mag) != 2 or len(distance) != 2:
        raise ValueError("theta, phi, v_mag, distance must all have length 2 (for B and C)")

    # --------------------------
    # Compute critical velocity
    # --------------------------
    v_crit = critical_velocity([50.0] * 6, sep, ecc)
    if hasattr(v_crit, "value_in"):  # AMUSE Quantity
        v_crit_val = float(v_crit.value_in(units.kms))
    else:
        v_crit_val = float(v_crit)

    # --------------------------
    # Binary A: fixed at origin
    # --------------------------
    centers = [[0.0, 0.0, 0.0]]
    v_vectors = [[0.0, 0.0, 0.0]]
    directions = [0.0]
    orbit_plane = [[0.0, 0.0, 1.0]]  # xy-plane
    phases = [true_anomalies[0]]

    # --------------------------
    # Binaries B and C
    # --------------------------
    for i in range(2):
        r = np.array(
            [
                distance[i] * np.sin(phi[i]) * np.cos(theta[i]),
                distance[i] * np.sin(phi[i]) * np.sin(theta[i]),
                distance[i] * np.cos(phi[i]),
            ]
        )
        # velocity points roughly toward origin
        v_dir = -r / np.linalg.norm(r)
        v_vec = v_mag[i] * v_crit_val * v_dir
        centers.append(r.tolist())
        v_vectors.append(v_vec.tolist())

        # Orbit plane
        n = np.cross(v_vec, r)
        n_norm = np.linalg.norm(n)
        n_unit = n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        orbit_plane.append(n_unit.tolist())

        # Direction in plane
        ref = np.array([1.0, 0.0, 0.0])
        v_proj = v_vec - np.dot(v_vec, n_unit) * n_unit
        if np.linalg.norm(v_proj) < 1e-12:
            angle = 0.0
        else:
            v_proj /= np.linalg.norm(v_proj)
            angle = np.arctan2(np.dot(np.cross(ref, v_proj), n_unit), np.dot(ref, v_proj))
        directions.append(float(angle))

        # Phase
        phases.append(true_anomalies[i + 1])

    return centers, v_vectors, directions, orbit_plane, phases
