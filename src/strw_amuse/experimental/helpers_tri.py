"""
==EXPERIMENTAL== General helper functions for AMUSE simulation on triplet case.
"""

import numpy as np
from amuse.datamodel import Particle
from amuse.units.quantities import VectorQuantity
from amuse.ext.orbital_elements import generate_binaries


def new_N_systems(params_bin_arr, params_trip_arr, iter):
    binaries = []
    triplets = []
    if len(params_bin_arr) == 0:
        for i in range(n_trip):
            trips = make_triplet(
                params_trip_arr[iter][1][i],
                params_trip_arr[iter][3][i],
                params_trip_arr[iter][2][i],
                params_trip_arr[iter][4][i],
                params_trip_arr[iter][5][i],
                params_trip_arr[iter][6][i],
            )
            triplets.append(trips)
        return binaries, triplets
    elif len(params_trip_arr) == 0:
        for i in range(n_bin):
            bins = make_binary_system(
                params_bin_arr[iter][1][i],
                params_bin_arr[iter][3][i],
                params_bin_arr[iter][2][i],
                params_bin_arr[iter][4][i],
                params_bin_arr[iter][5][i],
                params_bin_arr[iter][6][i],
            )
            binaries.append(bins)
        return binaries, triplets
    elif (len(params_bin_arr) != 0) & (len(params_trip_arr) != 0):
        for i in range(n_bin):
            bins = make_binary_system(
                params_bin_arr[iter][1][i],
                params_bin_arr[iter][3][i],
                params_bin_arr[iter][2][i],
                params_bin_arr[iter][4][i],
                params_bin_arr[iter][5][i],
                params_bin_arr[iter][6][i],
            )
            binaries.append(bins)
        for i in range(n_trip):
            trips = make_triplet(
                params_trip_arr[iter][1][i],
                params_trip_arr[iter][3][i],
                params_trip_arr[iter][2][i],
                params_trip_arr[iter][4][i],
                params_trip_arr[iter][5][i],
                params_trip_arr[iter][6][i],
            )
            triplets.append(trips)
        return binaries, triplets


def initiator(runs, array):
    params_bin_arr, params_trip_arr = vector_params(runs, array)
    for i in range(runs):
        binaries, triplets = new_N_systems(params_bin_arr, params_trip_arr, i)
        # result=run_simulation(binaries,triplets)
        # analyze(result)
    pass


def vector_params_bin(
    runs,
    n_bin,
    n_trip,
    m_min,
    m_max,
    vel_min,
    vel_max,
    position_min,
    position_max,
    sep_min,
    sep_max,
    ecc_min,
    ecc_max,
    phase_min,
    phase_max,
    theta_min,
    theta_max,
    phi_min,
    phi_max,
    imp_min,
    imp_max,
    psi_min,
    psi_max,
):
    masses = np.random.uniform(m_min, m_max, size=(runs, 6))

    eccs = np.random.uniform(ecc_min, ecc_max, size=(runs, 3))
    velocities = np.random.uniform(vel_min, vel_max, size=(runs, 2))
    position = np.random.uniform(position_min, position_max, size=(runs, 2))
    separations = np.random.uniform(sep_min, sep_max, size=(runs, 3))

    anomalies = np.random.uniform(phase_min, phase_max, size=(runs, 3))
    thetas = np.random.uniform(theta_min, theta_max, size=(runs, 2))

    phis = np.random.uniform(phi_min, phi_max, size=(runs, 2))
    impact = np.random.uniform(imp_min, imp_max, size=(runs, 2))
    psis = np.random.uniform(psi_min, psi_max, size=(runs, 2))

    # params_bin_arr=np.empty(runs,dtype='object')
    # params_trip_arr=np.empty(runs,dtype='object')
    # mass_select=np.random.choice(masses,size=2*n_bin,replace=False)
    # vel_select=np.random.choice(velocities,size=((n_bin-1),3),replace=False)
    # position_select=np.random.choice(position,size=((n_bin-1),3),replace=False)
    # separation_select=np.random.choice(separations,size=n_bin,replace=False)
    # eccs_select=np.random.choice(eccs,size=n_bin,replace=False)
    # phases_select=np.random.choice(phases,size=n_bin,replace=False)

    # mass_trip_select=np.random.choice(masses,size=3*n_trip,replace=False)
    # mass_trip_select=np.reshape(mass_trip_select,(n_trip,3))

    # vel_trip_select=np.random.choice(velocity_trip,size=9*n_trip,replace=False)
    # vel_trip_select=np.reshape(vel_trip_select,(3*n_trip,3))

    # position_trip_select=np.random.choice(position_trip,size=3*n_trip,replace=False)
    # position_trip_select=np.reshape(position_trip_select,(n_trip,3))

    # separation_trip_select=np.random.choice(separation,size=6*n_trip,replace=True)
    # separation_trip_select=np.reshape(separation_trip_select,(2*n_trip,3))

    # eccs_trip_select=np.random.choice(eccs,size=2*n_trip,replace=True)
    # eccs_trip_select=np.reshape(eccs_trip_select,(n_trip,2))

    # phases_trip_select=np.random.choice(phases,size=2*n_trip,replace=False)
    # phases_trip_select=np.reshape(phases_trip_select,(n_trip,2))
    # params_bin_arr[i]=(n_bin,mass_bin_select,vel_bin_select,position_bin_select,separation_bin_select,eccs_bin_select,phases_bin_select)
    # params_trip_arr[i]=(n_trip,mass_trip_select,vel_trip_select,position_trip_select,separation_trip_select,eccs_trip_select,phases_trip_select)

    return separations, eccs, velocities, phis, anomalies, thetas, impact, masses, position, psis


def vector_params_trip(
    runs,
    m_min,
    m_max,
    vel_min,
    vel_max,
    position_min,
    position_max,
    sep_min,
    sep_max,
    ecc_min,
    ecc_max,
    phase_min,
    phase_max,
    theta_min,
    theta_max,
    phi_min,
    phi_max,
    imp_min,
    imp_max,
    psi_min,
    psi_max,
):
    # Assumed that there are 2 trinaries, one is located at the center

    masses = np.random.uniform(m_min, m_max, size=(runs, 6))

    eccs = np.random.uniform(ecc_min, ecc_max, size=(runs, 4))
    velocities = np.random.uniform(vel_min, vel_max, size=(runs, 1))
    position = np.random.uniform(position_min, position_max, size=(runs, 1))
    separations = np.random.uniform(sep_min, sep_max, size=(runs, 4))

    anomalies = np.random.uniform(phase_min, phase_max, size=(runs, 4))
    thetas = np.random.uniform(theta_min, theta_max, size=(runs, 1))

    phis = np.random.uniform(phi_min, phi_max, size=(runs, 1))
    impact = np.random.uniform(imp_min, imp_max, size=(runs, 1))
    psis = np.random.uniform(psi_min, psi_max, size=(runs, 1))
    return separations, eccs, velocities, phis, anomalies, thetas, impact, masses, position, psis


def make_triplet(
    m1,
    m2,
    m3,
    a1,
    a2,
    e1=0.0,
    e2=0.0,
    center=None,
    direction=0.0,
    orbit_plane=[0, 0, 1],
    impact_parameter=0.0 | units.AU,
    f_a=0.0,  # true anomaly
    f_b=0.0,
    psi=0.0,  # impact orientation
):
    "Function takes list of masses, must be size 3, also takes a 3x3 array of positions and velocities"
    "Stars are initialized to the position of the star at position 0, so take this into account"

    plane = np.array(orbit_plane, dtype=float)
    plane /= np.linalg.norm(plane)

    # Reference vector perpendicular to orbit plane
    ref = np.array([1, 0, 0]) if abs(np.dot(plane, [1, 0, 0])) < 0.9 else np.array([0, 1, 0])
    off_set_dir = np.cross(plane, ref)
    off_set_dir /= np.linalg.norm(off_set_dir)
    c2, s2 = np.cos(direction + psi), np.sin(direction + psi)
    r_dir = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    off_set_dir = np.dot(r_dir, off_set_dir)

    # Masses
    m1 = m1 | units.MSun
    m2 = m2 | units.MSun
    m3 = m3 | units.MSun
    total_mass = m1 + m2 + m3
    # Center including impact parameter along offset
    if not isinstance(center, VectorQuantity):
        center = VectorQuantity(center, units.AU)
    center += off_set_dir * impact_parameter

    # Positions using true anomaly f
    r_rel_1 = a1 * (1 - e1**2) / (1 + e1 * np.cos(f_a))
    r1_2 = -(m2 / total_mass) * r_rel_1
    r2_1 = (m1 / total_mass) * r_rel_1

    r_rel_2 = a2 * (1 - e2**2) / (1 + e2 * np.cos(f_b))
    r3 = ((m1 + m2) / total_mass) * r_rel_2
    # r1_2 = -(m3 / total_mass) * r_rel_2
    M12 = m1 + m2
    # Orbital velocities
    if e1 == 0.0:
        v_rel_1 = (constants.G * M12 / a1) ** 0.5
    elif e1 < 1.0:
        v_rel_1 = (constants.G * M12 * (1 + e1) / (a1 * (1 - e1))) ** 0.5
    else:
        raise ValueError("Eccentricity (1) must be < 1")
    if e2 == 0.0:
        v_rel_2 = (constants.G * total_mass / a2) ** 0.5
    elif e2 < 1.0:
        v_rel_2 = (constants.G * total_mass * (1 + e2) / (a2 * (1 - e2))) ** 0.5
    else:
        raise ValueError("Eccentricity (2) must be < 1")

    v1_1 = +(m2 / M12) * v_rel_1
    v2_1 = -(m1 / M12) * v_rel_1
    v3 = -(m3 / total_mass) * v_rel_1
    v_cm = +(m3 / total_mass) * v_rel_2
    v3 = -(M12 / total_mass) * v_rel_2
    v1 = v1_1 + v_cm
    v2 = v2_1 + v_cm

    pos1_base = np.array(
        [r1_2.value_in(units.AU) * np.cos(f_a), r1_2.value_in(units.AU) * np.sin(f_a), 0.0]
    )
    pos2_base = np.array(
        [r2_1.value_in(units.AU) * np.cos(f_a), r2_1.value_in(units.AU) * np.sin(f_a), 0.0]
    )
    pos3_base = np.array(
        [r3.value_in(units.AU) * np.cos(f_b), r3.value_in(units.AU) * np.sin(f_b), 0.0]
    )
    vel1_base = np.array(
        [-v1.value_in(units.kms) * np.sin(f_a), v1.value_in(units.kms) * np.cos(f_a), 0.0]
    )
    vel2_base = np.array(
        [-v2.value_in(units.kms) * np.sin(f_a), v2.value_in(units.kms) * np.cos(f_a), 0.0]
    )
    vel3_base = np.array(
        [-v3.value_in(units.kms) * np.sin(f_b), v3.value_in(units.kms) * np.cos(f_b), 0.0]
    )
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
    pos3 = np.dot(r_total, pos3_base) | units.AU
    vel1 = np.dot(r_total, vel1_base) | units.kms
    vel2 = np.dot(r_total, vel2_base) | units.kms
    vel3 = np.dot(r_total, vel3_base) | units.kms

    p1 = Particle(mass=m1)
    p2 = Particle(mass=m2)
    p3 = Particle(mass=m3)
    p1.position = center + pos1
    p2.position = center + pos2
    p3.position = center + pos3
    p1.velocity = vel1
    p2.velocity = vel2
    p3.velocity = vel3
    return p1, p2, p3


def critical_velocity_trip(masses, sep, ecc):
    """
    Calculate the critical velocity for a triplet–tripletencounter.
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
    # === <<- section in between can be vectorized
    m1, m2, m3, m4, m5, m6 = masses
    a_1, a_2, b_1, b_2 = sep
    e_a1, e_a2, e_b1, e_b2 = ecc

    mu_a = (m1 * m2 * m3) / (m1 + m2 + m3)
    mu_b = (m4 * m5 * m6) / (m4 + m5 + m6)

    # === <<- section in between can be vectorized

    energy_a = (-G * m1 * m2 / (2 * a_1 * (1 - e_a1**2))) + (
        -G * (m1 + m2) * m3 / (2 * a_2 * (1 - e_a2**2))
    )
    energy_b = (-G * m4 * m5 / (2 * b_1 * (1 - e_b1**2))) + (
        -G * (m4 + m5) * m6 / (2 * b_2 * (1 - e_b2**2))
    )

    total_energy = energy_a + energy_b
    reduced_mass = (mu_a) / (mu_a + mu_b)

    v_crit = np.sqrt(-2 * total_energy / reduced_mass)  # in km/s

    return v_crit


from amuse.units import units, constants
from amuse.datamodel import Particles


def transformation_to_cartesian_trip(
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

    # --------------------------
    # Compute critical velocity
    # --------------------------
    v_crit = critical_velocity_trip([50.0] * 6, sep, ecc)
    if hasattr(v_crit, "value_in"):  # AMUSE Quantity
        v_crit_val = float(v_crit.value_in(units.kms))
    else:
        v_crit_val = float(v_crit)

    # --------------------------
    # Triplet A: fixed at origin
    # --------------------------
    centers = [[0.0, 0.0, 0.0]]
    v_vectors = [[0.0, 0.0, 0.0]]
    directions = [0.0]
    orbit_plane = [[0.0, 0.0, 1.0]]  # xy-plane
    phases = [true_anomalies[0], true_anomalies[1]]

    # --------------------------
    # Triplet B
    # --------------------------

    r = np.array(
        [
            distance[0] * np.sin(phi[0]) * np.cos(theta[0]),
            distance[0] * np.sin(phi[0]) * np.sin(theta[0]),
            distance[0] * np.cos(phi[0]),
        ]
    )
    # velocity points roughly toward origin
    v_dir = -r / np.linalg.norm(r)
    v_vec = v_mag[0] * v_crit_val * v_dir
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
    phases.append(true_anomalies[2])
    phases.append(true_anomalies[3])

    return centers, v_vectors, directions, orbit_plane, phases


def make_binary_system(masses, positions, velocities, seperations, ecc, phase):
    binary = generate_binaries(
        masses[0] | units.MSun,
        masses[1] | units.MSun,
        seperations[0] | units.AU,
        ecc[0],
        phase[0] | units.deg,
    )
    binary[0].position = positions | units.au
    binary[1].position += binary[0].position
    binary[0].velocities = velocities[0] | units.kms
    binary[1].velocities = velocities[1] | units.kms
    binary[1].velocities += binary[0].velocities
    return binary
