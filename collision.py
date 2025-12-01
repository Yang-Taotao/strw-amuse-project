"""
General helper functions for AMUSE simulation.
"""

# import

import numpy as np
import os

from amuse.datamodel.particle_attributes import bound_subset
from amuse.units import units, constants, nbody_system
from amuse.community.fi.interface import Fi
from amuse.datamodel import Particle, Particles
from amuse.io import write_set_to_file  # , read_set_from_file




from src.strw_amuse.config import (
    OUTPUT_DIR_COLLISIONS,
    OUTPUT_DIR_COLLISION_DIAGNOSTICS
)


# func repo
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

def run_fi_collision(gas, t_end=0.1 | units.yr,
                     min_mass=1e-6 | units.MSun,
                     run_label="default"):
    """
    Run a Fi SPH collision for a set of particles.
    Writes diagnostics to OUTPUT_DIR_COLLISION_DIAGNOSTICS as an AMUSE file.
    Returns: (gas_out, diag_summary_dict)
    """

    # Filter low-mass particles
    gas = gas[gas.mass > min_mass]
    if len(gas) == 0:
        raise ValueError("All SPH particles filtered out due to low mass.")

    # Center SPH set at COM
    com_pos = gas.center_of_mass()
    com_vel = gas.center_of_mass_velocity()
    gas.position -= com_pos
    gas.velocity -= com_vel

    # Length scale for N-body conversion
    lengths = gas.position.lengths()
    length_scale = lengths.max()
    if length_scale < 1e-3 | units.AU:
        length_scale = 1e-3 | units.AU

    total_mass = gas.total_mass()
    converter = nbody_system.nbody_to_si(total_mass, length_scale)

    # --- Run Fi ---
    hydro = Fi(converter)
    hydro.gas_particles.add_particles(gas)
    hydro.parameters.timestep = 0.01 | units.yr
    hydro.parameters.verbosity = 2

    try:
        hydro.evolve_model(t_end)
    finally:
        gas_out = hydro.gas_particles.copy()
        hydro.stop()

    # Restore COM frame
    gas_out.position += com_pos
    gas_out.velocity += com_vel

    # --- Energies ---
    KE = gas_out.kinetic_energy()
    PE = gas_out.potential_energy()
    TE = KE + PE

    # --- Bound particles ---
    core = max(gas_out, key=lambda p: p.mass)
    bound_particles = bound_subset(
        gas_out,
        core=core,
        density_weighting_power=2,
        smoothing_length_squared=gas_out.h_smooth**2
    )
    bound_fraction = len(bound_particles) / len(gas_out)

    # Angular momentum of bound remnant
    m = bound_particles.mass.value_in(units.kg)
    r = bound_particles.position.value_in(units.m)
    v = bound_particles.velocity.value_in(units.m/units.s)

    L_vec = np.sum(m[:, None] * np.cross(r, v), axis=0) | (units.kg * units.m**2 / units.s)

    r_rel = bound_particles.position - bound_particles.center_of_mass()
    I = (bound_particles.mass * r_rel.lengths()**2).sum()
    omega = (L_vec.length() / I).in_(1 / units.s)

    # --- Write diagnostics ---
    diag_particle = Particle()

    diag_particle.initial_mass = total_mass
    diag_particle.R_scale = length_scale
    diag_particle.N_sph = len(gas)
    diag_particle.final_mass = gas_out.total_mass()
    diag_particle.KE = KE
    diag_particle.PE = PE
    diag_particle.TE = TE
    diag_particle.bound_fraction = bound_fraction
    diag_particle.N_bound = len(bound_particles)
    diag_particle.Lx = L_vec[0]
    diag_particle.Ly = L_vec[1]
    diag_particle.Lz = L_vec[2]
    diag_particle.spin = omega

    diag_particles = Particles()
    diag_particles.add_particle(diag_particle)

    # file name
    diag_filename = os.path.join(
        OUTPUT_DIR_COLLISION_DIAGNOSTICS,
        f"collision_diag_{run_label}.amuse"
    )

    write_set_to_file(diag_particles, diag_filename, "amuse", overwrite_file=True)


    return gas_out




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
            print(f"Collision aborted: keys {key_i},{key_j} not found.")
            return False, None

        # Build SPH initial conditions
        colliders = Particles()
        colliders.add_particle(p_i.copy())
        colliders.add_particle(p_j.copy())
        sph = make_sph_from_two_stars(colliders, n_sph_per_star=500)
        

        # --- Pre-collision COM and velocity ---
        pre_com_pos = (p_i.mass * p_i.position + p_j.mass * p_j.position) / (p_i.mass + p_j.mass)
        pre_com_vel = (p_i.mass * p_i.velocity + p_j.mass * p_j.velocity) / (p_i.mass + p_j.mass)

        # Center SPH
        sph.position -= sph.center_of_mass()
        sph.velocity -= sph.center_of_mass_velocity()
 
        # Save SPH initial state
        final_filename = os.path.join(
            OUTPUT_DIR_COLLISIONS,
            f"collision_{n_collision}_sph_input_{run_label}.amuse"
        )
        write_set_to_file(sph, final_filename, "amuse", overwrite_file=True)

        # Run Fi
        gas_out = run_fi_collision(sph, t_end=0.1 | units.yr, run_label=f"{run_label}_collision{n_collision}")
        print("Fi collision done")

        # --- Bound particle selection ---
        r = gas_out.position - gas_out.center_of_mass()
        v = gas_out.velocity - gas_out.center_of_mass_velocity()
        r_mag = r.lengths()
        m_total = gas_out.total_mass()
        phi_pot = -(constants.G * m_total) / (r_mag + (1 | units.RSun))
        e_spec = 0.5 * v.lengths()**2 + phi_pot
        bound_mask = e_spec.value_in(units.m**2 / units.s**2) < 0.0

        if not np.any(bound_mask):
            print("No bound particles found: destructive collision")
            return remove_colliders(gravity, seba, key_map, [key_i, key_j])

        bound_particles = gas_out[bound_mask]
        m_bound = bound_particles.total_mass()
        remnant_radius = (m_bound.value_in(units.MSun)**0.57) | units.RSun

        # --- Very small remnant → destructive ---
        if m_bound <= (5 | units.MSun):
            print("Very small remnant mass; destructive collision")
            return remove_colliders(gravity, seba, key_map, [key_i, key_j])

        # --- Compute remnant velocity preserving SPH internal motion ---
        sph_com_vel_before_shift = gas_out.center_of_mass_velocity()
        remnant_vel = pre_com_vel + (bound_particles.center_of_mass_velocity() - sph_com_vel_before_shift)

        # --- Replace p_i with remnant properties ---
        # Compute remnant properties
        remnant_mass = m_bound
        remnant_radius = remnant_radius
        remnant_pos = pre_com_pos
        remnant_vel = pre_com_vel + (bound_particles.center_of_mass_velocity() - gas_out.center_of_mass_velocity())

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


        print(f"Collision {n_collision} processed: remnant = {m_bound.value_in(units.MSun):.2f} Msun, "
            f"(replacing {key_i}, removed {key_j})")

        return True, p_i


    except Exception as e:
        print("Collision handling failed with exception type:", type(e))
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

