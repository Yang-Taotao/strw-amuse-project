import numpy as np


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from amuse.units import units

from amuse.lab import *
from amuse.io import write_set_to_file, read_set_from_file

#import libraries
from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles

from itertools import combinations


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
        if not np.isfinite(radii[i].value_in(units.RSun)) or not np.isfinite(radii[j].value_in(units.RSun)):
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
    sph.mass = (mass / n_particles)

    # sample radius uniformly in volume (keep units)
    # convert radius to meters for numpy sampling then reattach unit
    r_vals = (radius.value_in(units.m) * np.random.random(n_particles)**(1/3)) | units.m
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
    sph.vx = 0. | units.kms
    sph.vy = 0. | units.kms
    sph.vz = 0. | units.kms

    # internal energy estimate
    if u_value is None:
        # virial-ish estimate: u ~ 0.2 * G M / R  (units J/kg)
        u_est = 0.2 * (constants.G * mass / radius)
        sph.u = u_est
    else:
        sph.u = u_value

    # compute a mean inter-particle spacing in meters and set h_smooth to a safe fraction
    mean_sep = ( (4/3.0)*np.pi*(radius.value_in(units.m)**3) / n_particles )**(1/3) | units.m
    # choose smoothing length ~ 1.2 * mean_sep (safe number of neighbors)
    sph.h_smooth = (1.2 * mean_sep).in_(pos_unit)

    return sph


def make_sph_from_two_stars(stars, n_sph_per_star=100, u_value=None, pos_unit=units.AU):
    if len(stars) != 2:
        raise ValueError("Expect exactly two stars")

    s1, s2 = stars[0], stars[1]

    sph1 = create_sph_star(s1.mass, s1.radius, n_particles=n_sph_per_star, u_value=u_value, pos_unit=pos_unit)
    sph2 = create_sph_star(s2.mass, s2.radius, n_particles=n_sph_per_star, u_value=u_value, pos_unit=pos_unit)

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
        "N": len(gas)
    }

    return gas_out, diagnostics


def compute_remnant_spin(gas):
    """
    Compute mass, COM velocity, spin omega, and angular momentum of remnant.
    """
    COM_pos = gas.center_of_mass()
    COM_vel = gas.center_of_mass_velocity()
    L = VectorQuantity([0.0, 0.0, 0.0], units.kg * units.m**2 / units.s)
    I_scalar = 0. | units.kg * units.m**2

    for p in gas:
        r = p.position - COM_pos
        v = p.velocity - COM_vel
        L += p.mass * r.cross(v)
        I_scalar += p.mass * r.length()**2

    omega = (L.length() / I_scalar).in_(1/units.s)
    Mbound = gas.total_mass()
    Vcom = COM_vel.in_(units.kms)
    return Mbound, Vcom, omega, L

def make_triple_binary_system(
    masses,
    seps,
    ecc,
    directions,
    orbit_plane,
    impact_parameter,
    centers=None,
    v_coms=None
):
    """
    Create a system of three interacting binaries with fully tunable parameters.
    """

    if not (len(masses) == 6 and len(seps) == 3 and len(directions) == 3):
        raise ValueError("Expect masses=6, seps=3, directions=3.")

    # Default centers
    if centers is None:
        centers = [
            [-300, 0, 0],
            [300, 0, 0],
            [0, 600, 0]
        ]
    # Default COM velocities
    if v_coms is None:
        v_coms = [
            [10., 0., 0.],
            [-10., 0., 0.],
            [0., -10., 0.]
        ]

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
    p1, p2 = make_binary(ma1, ma2, sepA | units.AU, eccA, center=centerA, direction=dirA, orbit_plane = orbit_plane[0], impact_parameter=0.0)
    p3, p4 = make_binary(mb1, mb2, sepB | units.AU, eccB, center=centerB, direction=dirB,orbit_plane = orbit_plane[1], impact_parameter=impact_parameter[0])
    p5, p6 = make_binary(mc1, mc2, sepC | units.AU, eccC, center=centerC, direction=dirC,orbit_plane = orbit_plane[2], impact_parameter=impact_parameter[1])

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



def make_binary(m1, m2, a, e=0.0, center=None, direction=0.0, orbit_plane=[0,0,1], impact_parameter=0.0):
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
        Rotation angle around orbit normal vector (radians).
    orbit_plane : list of 3 floats
        Normal vector defining orbital plane. Default: z-axis.

    Returns
    -------
    p1, p2 : Particle
        Two AMUSE particles with positions and velocities.
    """
    # direction is a rotation angle (radians)
    plane = np.array(orbit_plane, dtype=float)
    plane /= np.linalg.norm(plane)

    # Choose a reference axis perpendicular to orbit normal (e.g., x-axis if not parallel)
    ref = np.array([1, 0, 0]) if abs(np.dot(plane, [1, 0, 0])) < 0.9 else np.array([0, 1, 0])
    off_set_dir = np.cross(plane, ref)
    off_set_dir /= np.linalg.norm(off_set_dir)
    # Rotate offset direction by given angle around plane normal
    c2, s2 = np.cos(direction), np.sin(direction)
    R_dir = np.array([[c2, -s2, 0],
                    [s2,  c2, 0],
                    [0,   0,  1]])
    off_set_dir = np.dot(R_dir, off_set_dir)



    m1 = m1 | units.MSun
    m2 = m2 | units.MSun
    total_mass = m1 + m2
    if impact_parameter is None:
        impact_parameter = 0.0 | units.AU
    else:
        impact_parameter = impact_parameter | units.AU  
    # Default center
    if center is None:
        center = VectorQuantity([0,0,0], units.AU)
        #offset centers by impact parameter along offset direction
        center = center + off_set_dir * impact_parameter
    elif not isinstance(center, VectorQuantity):
        center = VectorQuantity(center, units.AU)
        center = center + off_set_dir * impact_parameter

    # Separation at pericenter
    r_rel = a * (1 - e)
    r1 = -(m2 / total_mass) * r_rel
    r2 =  (m1 / total_mass) * r_rel

    # Circular or elliptical orbit velocity
    if e == 0.0:
        v_rel = (constants.G * total_mass / a)**0.5
    elif e < 1.0:
        v_rel = ((constants.G * total_mass * (1 + e) / (a * (1 - e)))**0.5)
    else:
        raise ValueError("Eccentricity must be < 1")

    v1 = + (m2 / total_mass) * v_rel
    v2 = - (m1 / total_mass) * v_rel

    # Base positions and velocities in XY plane
    pos1_base = np.array([r1.value_in(units.AU), 0., 0.])
    pos2_base = np.array([r2.value_in(units.AU), 0., 0.])
    vel1_base = np.array([0., v1.value_in(units.kms), 0.])
    vel2_base = np.array([0., v2.value_in(units.kms), 0.])

    # Normalize orbit_plane
    n = np.array(orbit_plane, dtype=float)
    n /= np.linalg.norm(n)

    # Rotation to align Z-axis with orbit normal
    z_axis = np.array([0., 0., 1.])
    v = np.cross(z_axis, n)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, n)

    if s == 0:  # already aligned
        R_plane = np.eye(3)
        if c < 0:  # opposite direction
            R_plane = -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_plane = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s**2))

    # Additional rotation around orbit normal
    c2, s2 = np.cos(direction), np.sin(direction)
    R_dir = np.array([[c2, -s2, 0],
                      [s2,  c2, 0],
                      [0,   0,  1]])

    # Total rotation
    R_total = np.dot(R_plane, R_dir)

    pos1 = np.dot(R_total, pos1_base) | units.AU
    pos2 = np.dot(R_total, pos2_base) | units.AU
    vel1 = np.dot(R_total, vel1_base) | units.kms
    vel2 = np.dot(R_total, vel2_base) | units.kms

    # Create particles
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
    seba = SeBa()   # fast SSE-style stellar evolution
    stars = Particles()
    for m in masses_msun:
        p = Particle(mass = m | units.MSun)
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
    G = constants.G.in_(units.AU**3 / (units.MSun * units.day**2)) # AU^3 / (MSun * day^2)
    G = G.value_in(units.kms**2 * units.AU / units.MSun)  # km^2/s^2 * AU / MSun
    m1, m2, m3, m4, m5, m6 = masses
    a1, a2, a3 = sep
    e1, e2, e3 = ecc

    mu1 = (m1 * m2) / (m1 + m2)
    mu2 = (m3 * m4) / (m3 + m4)
    mu3 = (m5 * m6) / (m5 + m6)

    E1 = -G * m1 * m2 / (2 * a1 * (1 - e1**2))
    E2 = -G * m3 * m4 / (2 * a2 * (1 - e2**2))
    E3 = -G * m5 * m6 / (2 * a3 * (1 - e3**2))

    total_energy = E1 + E2 + E3
    reduced_mass = (mu1 * mu2) / (mu1 + mu2 + mu3)

    v_crit = np.sqrt(-2 * total_energy / reduced_mass)  # in km/s


    return v_crit

def outcomes(initial_particles, final_particles, max_mass=None):
    """
    Classify the end result of the simulation based on the number of surviving stars
    and the presence (or absence) of a massive merger remnant.

    Outcome classes:
    - 'no_collision' : No stars merged or destroyed.
    - 'destructive'  : One or more collisions occurred, no massive remnant.
    - 'creative_ionized' : One or more collisions occurred; remnant is unbound.
    - 'creative_bound'   : One or more collisions occurred; remnant is in a bound system.

    Returns
    -------
    tuple
        (outcome_label:str, descriptive_text:str)
    """
    from amuse.units import units, constants

    # --- Helper function ---
    def is_bound_system(particles):
        """Return True if any pair of particles is gravitationally bound."""
        if len(particles) < 2:
            return False
        for i, p_i in enumerate(particles[:-1]):
            for j, p_j in enumerate(particles[i+1:], i+1):
                r_vec = p_i.position - p_j.position
                v_vec = p_i.velocity - p_j.velocity
                r = r_vec.length()
                v = v_vec.length()
                E_spec = 0.5 * v**2 - constants.G * (p_i.mass + p_j.mass) / r
                if E_spec < 0 | (units.m**2 / units.s**2):
                    return True
        return False

    n_init = len(initial_particles)
    n_final = len(final_particles)

    # --- 1. No collision ---
    if n_init == n_final and max_mass is not None:
        label = 'no_collision'
        text = "No collision — all stars survived with their original masses."
        return label, text

    # --- 2. Destructive collision ---
    if max_mass is None:
        label = 'destructive'
        text = "Destructive collision — one or more stars destroyed, no remnant formed."
        return label, text

    # --- 3. Creative collision ---
    if n_final < n_init:
        bound = is_bound_system(final_particles)
        if bound:
            label = 'creative_bound'
            if max_mass < 80 | units.MSun:
                text = "Creative collision — remnant < 80 M☉ in bound system."
            else:
                text = (f"Creative collision — bound system formed with remnant "
                        f"of {max_mass.value_in(units.MSun):.1f} M☉.")
        else:
            label = 'creative_ionized'
            text = (f"Creative collision — ionized outcome, isolated remnant "
                    f"of {max_mass.value_in(units.MSun):.1f} M☉.")
        return label, text

    # --- Fallback ---
    label = 'unclassified'
    text = "Unclassified outcome — ambiguous particle count or data inconsistency."
    return label, text



