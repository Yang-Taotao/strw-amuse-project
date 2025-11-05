"""
Helper functions for stellar collisions and SPH setup in AMUSE.
"""

# import
from itertools import combinations
import numpy as np

from amuse.units import units
from amuse.lab import *
from amuse.community.seba.interface import SeBa
from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles

# func repo


def pairwise_separations(particles):
    """Return list of (i,j, separation_length) for particle set."""
    pairs = []
    for i, j in combinations(range(len(particles)), 2):
        sep = (particles[i].position - particles[j].position).length()
        pairs.append((i, j, sep))
    return pairs


def detect_close_pair(particles, radii, buffer_factor=0.4):
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
    """Create SPH representation of two stars."""
    if len(stars) != 2:
        raise ValueError("Expect exactly two stars")

    s1, s2 = stars[0], stars[1]

    sph1 = create_sph_star(
        mass=s1.mass,
        radius=s1.radius,
        n_particles=n_sph_per_star,
        u_value=u_value,
        pos_unit=pos_unit,
    )
    sph2 = create_sph_star(
        mass=s2.mass,
        radius=s2.radius,
        n_particles=n_sph_per_star,
        u_value=u_value,
        pos_unit=pos_unit,
    )

    # shift to absolute positions
    sph1.position += s1.position.in_(pos_unit)
    sph2.position += s2.position.in_(pos_unit)

    sph1.velocity += s1.velocity
    sph2.velocity += s2.velocity

    gas = Particles()
    gas.add_particles(sph1)
    gas.add_particles(sph2)

    return gas


def run_fi_collision(gas, t_end=0.1 | units.yr, min_mass=1e-6 | units.MSun):
    """Handles collision via Fi SPH code."""
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
        "M": total_mass.in_(units.MSun),
        "Rscale": length_scale.in_(units.AU),
        "N": len(gas),
    }

    return gas_out, diagnostics


def compute_remnant_spin(gas):
    """
    Compute mass, COM velocity, spin omega, and angular momentum of remnant.
    """
    COM_pos = gas.center_of_mass()
    COM_vel = gas.center_of_mass_velocity()
    L = VectorQuantity([0.0, 0.0, 0.0], units.kg * units.m**2 / units.s)
    I_scalar = 0.0 | units.kg * units.m**2

    for p in gas:
        r = p.position - COM_pos
        v = p.velocity - COM_vel
        L += p.mass * r.cross(v)
        I_scalar += p.mass * r.length() ** 2

    omega = (L.length() / I_scalar).in_(1 / units.s)
    Mbound = gas.total_mass()
    Vcom = COM_vel.in_(units.kms)
    return Mbound, Vcom, omega, L


def make_triple_binary_system(masses, seps, ecc, directions, centers=None, v_coms=None):
    """
    Create a system of three interacting binaries with fully tunable parameters.
    """

    if not (len(masses) == 6 and len(seps) == 3 and len(directions) == 3):
        raise ValueError("Expect masses=6, seps=3, directions=3.")

    # Default centers
    if centers is None:
        centers = [[-300, 0, 0], [300, 0, 0], [0, 600, 0]]
    # Default COM velocities
    if v_coms is None:
        v_coms = [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.0, -10.0, 0.0]]

    ma1, ma2, mb1, mb2, mc1, mc2 = masses
    sepA, sepB, sepC = seps
    dirA, dirB, dirC = directions
    eccA, eccB, eccC = ecc

    # Convert centers and velocities to VectorQuantity with units
    centerA = VectorQuantity(centers[0], units.AU)
    centerB = VectorQuantity(centers[1], units.AU)
    centerC = VectorQuantity(centers[2], units.AU)

    v_com_A = VectorQuantity(v_coms[0], units.kms)
    v_com_B = VectorQuantity(v_coms[1], units.kms)
    v_com_C = VectorQuantity(v_coms[2], units.kms)

    # Create binaries
    p1, p2 = make_binary(
        ma1, ma2, sepA | units.AU, eccA, center=centerA, direction=dirA
    )
    p3, p4 = make_binary(
        mb1, mb2, sepB | units.AU, eccB, center=centerB, direction=dirB
    )
    p5, p6 = make_binary(
        mc1, mc2, sepC | units.AU, eccC, center=centerC, direction=dirC
    )

    # Name particles
    p1.name, p2.name = "A1", "A2"
    p3.name, p4.name = "B1", "B2"
    p5.name, p6.name = "C1", "C2"

    # Apply COM velocities
    for p in (p1, p2):
        p.velocity += v_com_A
    for p in (p3, p4):
        p.velocity += v_com_B
    for p in (p5, p6):
        p.velocity += v_com_C

    # Combine all particles
    particles = Particles()
    for p in [p1, p2, p3, p4, p5, p6]:
        particles.add_particle(p)

    return particles


def make_binary(m1, m2, a, e=0.0, center=None, direction=0.0, orbit_plane=None):
    """
    Create a binary system with arbitrary eccentricity and orientation.

    Parameters
    ----------
    m1, m2 : float or Quantity
        Masses in Msun.
    a : Quantity
        Semi-major axis (AU).
    e : float
        Eccentricity (0=circular, 0<e<1=elliptical).
    center : VectorQuantity or list
        Center-of-mass position (default: [0,0,0] AU).
    direction : float
        Rotation angle around z-axis (radians) to orient the orbit.
    orbit_plane : list of 3 floats
        Normal vector defining orbital plane. Default: z-axis.

    Returns
    -------
    p1, p2 : Particle
        Two AMUSE particles with positions and velocities.
    """

    # set orbit plane to default z-axis if none provided
    if orbit_plane is None:
        orbit_plane = [0, 0, 1]

    # units assignment
    m1 = m1 | units.MSun
    m2 = m2 | units.MSun
    total_mass = m1 + m2

    # Default center
    if center is None:
        center = VectorQuantity(array=[0, 0, 0], unit=units.AU)
    elif not isinstance(center, VectorQuantity):
        center = VectorQuantity(array=center, unit=units.AU)

    # Circular approximation if e=0
    # More generally, sample at pericenter for simplicity
    r_rel = a * (1 - e)  # separation at pericenter
    r1 = -(m2 / total_mass) * r_rel
    r2 = (m1 / total_mass) * r_rel

    # Rotation matrix around z (or orbit_plane)
    c, s = np.cos(direction), np.sin(direction)
    R = np.array(object=[[c, -s, 0], [s, c, 0], [0, 0, 1]])

    pos1 = np.dot(a=R, b=[r1.value_in(units.AU), 0.0, 0.0]) | units.AU
    pos2 = np.dot(a=R, b=[r2.value_in(units.AU), 0.0, 0.0]) | units.AU

    p1 = Particle(mass=m1)
    p2 = Particle(mass=m2)
    p1.position = center + pos1
    p2.position = center + pos2

    # Circular or elliptical orbit velocity
    if e == 0.0:
        # circular orbit
        v_rel = (constants.G * total_mass / a) ** 0.5
    elif e < 1.0:
        # elliptical
        v_rel = (constants.G * total_mass * float(1 + e) / (a * float(1 - e))) ** 0.5
    else:
        raise ValueError("Eccentricity cannot be > or = 1")

    v1 = +(m2 / total_mass) * v_rel
    v2 = -(m1 / total_mass) * v_rel

    vel1 = np.dot(a=R, b=[0.0, v1.value_in(units.kms), 0.0]) | units.kms
    vel2 = np.dot(a=R, b=[0.0, v2.value_in(units.kms), 0.0]) | units.kms
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
        stars.add_particle(particle=p)

    seba.particles.add_particles(stars)
    seba.evolve_model(age)

    return seba, seba.particles
