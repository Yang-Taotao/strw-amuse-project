"""
Stopping condition utilities for 6-body encounter simulation.
"""

# --- helper functions (local) ---
import os

from amuse.datamodel import Particle, Particles
from amuse.datamodel.particle_attributes import bound_subset
from amuse.io import write_set_to_file
from amuse.units import constants, units

from ..utils.config import OUTPUT_DIR_COLLISIONS_OUTCOMES


def specific_pair_energy(p_i, p_j):
    """Return the specific relative energy (per reduced mass) between two particles:
    E = 0.5 * v_rel^2 - G*(m_i + m_j)/r
    Positive -> unbound, Negative -> bound.
    """
    r_vec = p_i.position - p_j.position
    r = r_vec.length()
    v_vec = p_i.velocity - p_j.velocity
    v2 = v_vec.length_squared()
    # E has units of (velocity^2)
    E = 0.5 * v2 - constants.G * (p_i.mass + p_j.mass) / r
    return E


def is_ionized_single(p_index, particles):
    """Check whether particle at index p_index is unbound from every other particle."""
    p = particles[p_index]
    for j, q in enumerate(particles):
        if j == p_index:
            continue
        E = specific_pair_energy(p, q)
        # bound if E < 0
        if E.value_in(units.m**2 / units.s**2) < 0:
            return False
    return True


def find_bound_groups(particles):
    """Return list of lists: connected components where edges are pairwise-bound (E < 0).
    Heuristic: negative-energy pairs indicate bound link.
    """
    n = len(particles)
    # build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            E = specific_pair_energy(particles[i], particles[j])
            if E.value_in(units.m**2 / units.s**2) < 0:
                adj[i].append(j)
                adj[j].append(i)
    # connected components via DFS
    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    stack.append(v)
        components.append(comp)
    return components


def group_physical_size(particles, group_indices):
    """Estimate group size as maximum pairwise distance inside group."""
    if len(group_indices) <= 1:
        return 0 | units.AU
    maxd = 0 | units.AU
    for i_idx in range(len(group_indices)):
        for j_idx in range(i_idx + 1, len(group_indices)):
            i = group_indices[i_idx]
            j = group_indices[j_idx]
            d = (particles[i].position - particles[j].position).length()
            if d > maxd:
                maxd = d
    return maxd


def group_com(particles, group_indices):
    # If group is empty, return a zero position vector with units
    if len(group_indices) == 0:
        return particles[0].position * 0

    # Proper AMUSE-safe mass sum:
    total_m = 0 | units.kg
    for i in group_indices:
        total_m += particles[i].mass

    # Compute center of mass
    com_pos = particles[0].position * 0
    for idx in group_indices:
        p = particles[idx]
        com_pos += p.position * (p.mass / total_m)

    return com_pos


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
