"""
General helper functions for AMUSE simulation.
"""

# import
from itertools import combinations
import numpy as np

from amuse.units import units, constants, nbody_system
from amuse.units.quantities import VectorQuantity
from amuse.community.fi.interface import Fi
from amuse.community.seba.interface import SeBa
from amuse.datamodel import Particles, Particle


# func repo


def pairwise_separations(particles):
    """Return list of (i,j, separation_length) for particle set."""
    pairs = []
    for i, j in combinations(range(len(particles)), 2):
        sep = (particles[i].position - particles[j].position).length()
        pairs.append((i, j, sep))
    return pairs


def detect_close_pair(particles, radii, buffer_factor=2):
    """
    Detects the first close pair based on radii overlap with a simple buffer.
    Returns (i, j, sep) or None.

    For some reason after testing an ideal buffer should be 0.3<b<0.4.
    Idk why but you can test this yourself
    Higher values seem to give unphysical encounters.
    Smaller values might be possible
    """
    for i, j, sep in pairwise_separations(particles):
        # Skip invalid radii
        if not np.isfinite(radii[i].value_in(units.RSun)) or not np.isfinite(
            radii[j].value_in(units.RSun)
        ):
            continue
        if radii[i] <= 0 | units.RSun or radii[j] <= 0 | units.RSun:
            continue

        # Use a simple buffer multiplier
        threshold = (radii[i] + radii[j]) * buffer_factor
        if sep < threshold:
            return (i, j, sep)
    return None


def create_sph_star(mass, radius, n_particles=10000, u_value=None, pos_unit=units.AU):
    """
    Create a uniform-density SPH star with safer defaults.
    mass, radius: AMUSE quantities
    pos_unit: coordinate unit for output positions
    """
    sph = Particles(n_particles)

    # set per-particle mass (AMUSE broadcasts quantity)
    sph.mass = mass / n_particles

    # sample radius uniformly in volume (keep units)
    # convert radius to meters for numpy sampling then reattach unit
    r_vals = (
        radius.value_in(units.m) * np.random.random(n_particles) ** (1 / 3)
    ) | units.m
    theta = np.arccos(2.0 * np.random.random(n_particles) - 1.0)
    phi = 2.0 * np.pi * np.random.random(n_particles)

    x = r_vals * np.sin(theta) * np.cos(phi)
    y = r_vals * np.sin(theta) * np.sin(phi)
    z = r_vals * np.cos(theta)

    # attach coordinates in requested unit
    sph.x = x.in_(pos_unit)
    sph.y = y.in_(pos_unit)
    sph.z = z.in_(pos_unit)

    # velocities zero in star frame
    sph.vx = 0.0 | units.kms
    sph.vy = 0.0 | units.kms
    sph.vz = 0.0 | units.kms

    # internal energy estimate
    if u_value is None:
        # virial-ish estimate: u ~ 0.2 * G M / R  (units J/kg)
        u_est = 0.2 * (constants.G * mass / radius)
        sph.u = u_est
    else:
        sph.u = u_value

    # compute a mean inter-particle spacing in meters and set h_smooth to a safe fraction
    mean_sep = ((4 / 3.0) * np.pi * (radius.value_in(units.m) ** 3) / n_particles) ** (
        1 / 3
    ) | units.m
    # choose smoothing length ~ 1.2 * mean_sep (safe number of neighbors)
    sph.h_smooth = (1.2 * mean_sep).in_(pos_unit)

    return sph


def make_sph_from_two_stars(stars, n_sph_per_star=100, u_value=None, pos_unit=units.AU):
    if len(stars) != 2:
        raise ValueError("Expect exactly two stars")

    s1, s2 = stars[0], stars[1]

    sph1 = create_sph_star(
        s1.mass,
        s1.radius,
        n_particles=n_sph_per_star,
        u_value=u_value,
        pos_unit=pos_unit,
    )
    sph2 = create_sph_star(
        s2.mass,
        s2.radius,
        n_particles=n_sph_per_star,
        u_value=u_value,
        pos_unit=pos_unit,
    )

    # shift to absolute positions <<- ATTENTION - Check if your instance of Particles are fully initialized AMUSE obj
    sph1.position += s1.position.in_(pos_unit)
    sph2.position += s2.position.in_(pos_unit)

    sph1.velocity += s1.velocity
    sph2.velocity += s2.velocity

    gas = Particles()
    gas.add_particles(sph1)
    gas.add_particles(sph2)

    return gas


def run_fi_collision(gas, t_end=0.1 | units.yr, min_mass=1e-6 | units.MSun):
    gas = gas[gas.mass > min_mass]
    if len(gas) == 0:
        raise ValueError("All SPH particles filtered out due to low mass.")

    com_pos = gas.center_of_mass()
    com_vel = gas.center_of_mass_velocity()
    gas.position -= com_pos
    gas.velocity -= com_vel

    lengths = (gas.position).lengths()
    length_scale = lengths.max()
    if length_scale < 1e-3 | units.AU:
        length_scale = 1e-3 | units.AU

    total_mass = gas.total_mass()
    converter = nbody_system.nbody_to_si(total_mass, length_scale)

    hydro = Fi(converter)
    hydro.gas_particles.add_particles(gas)
    hydro.parameters.timestep = 0.01 | units.yr
    hydro.parameters.verbosity = 2

    try:
        hydro.evolve_model(t_end)
    except Exception as e:
        hydro.stop()
        raise RuntimeError("Fi crash inside evolve_model") from e

    gas_out = hydro.gas_particles.copy()
    hydro.stop()
    gas_out.position += com_pos
    gas_out.velocity += com_vel

    # Diagnostics to return instead of printing
    diagnostics = {
        "Initial Mass": total_mass.in_(units.MSun),
        "Rscale": length_scale.in_(units.AU),
        "N": len(gas),
    }

    return gas_out, diagnostics


def compute_remnant_spin(gas):
    """
    Compute mass, com velocity, spin omega, and angular momentum of remnant.
    """
    com_pos = gas.center_of_mass()
    com_vel = gas.center_of_mass_velocity()
    angular_momentum = VectorQuantity([0.0, 0.0, 0.0], units.kg * units.m**2 / units.s)
    moment_of_inertia = 0.0 | units.kg * units.m**2

    for p in gas:
        r = p.position - com_pos
        v = p.velocity - com_vel
        angular_momentum += p.mass * r.cross(v)
        moment_of_inertia += p.mass * r.length() ** 2

    omega = (angular_momentum.length() / moment_of_inertia).in_(1 / units.s)
    m_bound = gas.total_mass()
    v_com = com_vel.in_(units.kms)
    return m_bound, v_com, omega, angular_momentum


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
    orbit_plane=[0, 0, 1],
    impact_parameter=0.0 | units.AU,
    f=0.0,  # true anomaly
    psi=0.0,  # impact orientation
):
    """
    Create a binary system with arbitrary eccentricity, orbit plane, phase, and impact orientation.
    """

    plane = np.array(orbit_plane, dtype=float)
    plane /= np.linalg.norm(plane)

    # Reference vector perpendicular to orbit plane
    ref = (
        np.array([1, 0, 0])
        if abs(np.dot(plane, [1, 0, 0])) < 0.9
        else np.array([0, 1, 0])
    )
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
    Calculate the critical velocity for a binary–binary-binary encounter.
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
    G = constants.G.in_(
        units.AU**3 / (units.MSun * units.day**2)
    )  # AU^3 / (MSun * day^2)
    G = G.value_in(units.kms**2 * units.AU / units.MSun)  # km^2/s^2 * AU / MSun
    # === <<- section in between can be vectorized
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


from amuse.units import units, constants
from amuse.datamodel import Particles


def binding_energy(i, j):
    r = (i.position - j.position).length()
    v = (i.velocity - j.velocity).length()
    mu = (i.mass * j.mass) / (i.mass + j.mass)
    return 0.5 * mu * v**2 - constants.G * i.mass * j.mass / r


def compute_bound_components(particles):
    """Return connected gravitationally bound components."""
    N = len(particles)
    adjacency = {p.key: set() for p in particles}

    # Build adjacency list of bound pairs
    for a in range(N):
        for b in range(a + 1, N):
            p = particles[a]
            q = particles[b]
            if binding_energy(p, q) < (0 | units.J):
                adjacency[p.key].add(q.key)
                adjacency[q.key].add(p.key)

    # DFS to extract connected components
    visited = set()
    components = []
    for p in particles:
        if p.key in visited:
            continue
        stack = [p.key]
        comp = []
        while stack:
            k = stack.pop()
            if k not in visited:
                visited.add(k)
                comp.append(k)
                stack.extend(adjacency[k] - visited)
        components.append(comp)

    return components


def count_collisions(particle):
    """
    Recursively count how many merger events formed this particle.
    AMUSE merger products have attributes child1, child2.
    """
    n = 0
    stack = [particle]
    while stack:
        p = stack.pop()
        if hasattr(p, "child1") and p.child1 is not None:
            n += 1
            stack.append(p.child1)
        if hasattr(p, "child2") and p.child2 is not None:
            n += 1
            stack.append(p.child2)
    return n


def outcomes(
    initial_particles,
    final_particles,
    massive_threshold=70 | units.MSun,
    creative_threshold=10 | units.MSun,
):
    """
    Final outcome classifier (pickle-safe).
    Returns: (label: str, info: dict)
    """

    # Ensure Particles format
    if isinstance(final_particles, list):
        final_particles = Particles(final_particles)

    n_init = len(initial_particles)
    n_final = len(final_particles)

    # -------------------------------------------------------------
    # NO COLLISION
    # -------------------------------------------------------------
    if n_final == n_init:
        return "no_collision", {
            "description": "All stars survived.",
            "n_final": n_final,
        }

    # -------------------------------------------------------------
    # ONE COLLISION
    # -------------------------------------------------------------
    if n_final == n_init - 1:

        final_max_mass = max(p.mass for p in final_particles)
        mm = max(final_particles, key=lambda p: p.mass)

        if final_max_mass < creative_threshold:
            return "destructive", {
                "remnant_mass_Msun": final_max_mass.value_in(units.MSun)
            }

        if final_max_mass < massive_threshold:
            return "creative_not_massive", {
                "remnant_mass_Msun": final_max_mass.value_in(units.MSun)
            }

        # Massive remnant
        if len(final_particles) == 1:
            return "creative_ionized", {
                "remnant_mass_Msun": final_max_mass.value_in(units.MSun),
            }

        comps = compute_bound_components(final_particles)
        comp = next(c for c in comps if mm.key in c)

        if len(comp) > 1:
            return "creative_bound", {
                "remnant_mass_Msun": final_max_mass.value_in(units.MSun),
                "n_companions": len(comp) - 1,
            }
        else:
            return "creative_ionized", {
                "remnant_mass_Msun": final_max_mass.value_in(units.MSun),
            }

        # -------------------------------------------------------------
    # MULTIPLE COLLISIONS
    # -------------------------------------------------------------
    components = compute_bound_components(final_particles)
    massive_stars = final_particles.select(lambda m: m >= massive_threshold, ["mass"])

    if len(massive_stars) == 0:
        return "creative_not_massive", {
            "description": "Multiple mergers, no massive remnants.",
            "n_massive": 0,
        }

    summary = []

    for m in massive_stars:
        # count how many mergers built this object
        n_coll = count_collisions(m)

        # group membership
        comp = next(c for c in components if m.key in c)
        comp_size = len(comp)

        if comp_size == 1:
            rem_type = "creative_ionized"
        else:
            rem_type = "creative_bound"

        summary.append(
            {
                "type": rem_type,
                "mass_Msun": m.mass.value_in(units.MSun),
                "n_collisions": n_coll,
                "component_size": comp_size,
            }
        )

    return "multiple_collisions", summary


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
    incoming binaries B and C approaching roughly from opposite directions,
    using independent theta and phi for each binary.

    Parameters:
        theta : list of 2 floats, polar angles for B and C
        phi   : list of 2 floats, azimuthal angles for B and C
        v_mag : list of 2 floats, velocities in units of fraction of critical

    Returns:
        centers, v_vectors, directions, orbit_plane, phases
    """
    if len(theta) != 2 or len(phi) != 2 or len(v_mag) != 2 or len(distance) != 2:
        raise ValueError(
            "theta, phi, v_mag, distance must all have length 2 (for B and C)"
        )

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
            angle = np.arctan2(
                np.dot(np.cross(ref, v_proj), n_unit), np.dot(ref, v_proj)
            )
        directions.append(float(angle))

        # Phase
        phases.append(true_anomalies[i + 1])

    return centers, v_vectors, directions, orbit_plane, phases
