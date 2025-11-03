"""
Monte Carlo cross-section calculation for triple–triple encounters
(adapted from Hut & Bahcall 1983 setup)
using run_6_body_simulation().
"""

import numpy as np
import os
import time
from argparse import ArgumentParser

from amuse.lab import units, constants
from amuse.io import write_set_to_file

from run_6body_encounter import run_6_body_simulation
from helpers import make_triple_binary_system

# ======================================================
# Utility functions
# ======================================================

def sample_initial_conditions():
    # Masses (MSun)
    masses = np.random.uniform([10, 10, 10, 10, 10, 10], [100, 80, 60, 80, 60, 40])
    
    # Separations (AU)
    sep = np.random.uniform(0.1, 5.0, 3)
    
    # Eccentricities (0 <= e < 1)
    ecc = np.random.uniform(0.0, 0.9, 3)
    
    # Orbital directions (for inner binaries)
    directions = np.random.uniform(-1.0, 1.0, 3)
    
    
    
    return masses.tolist(), sep.tolist(), ecc.tolist(), directions.tolist()




def critical_velocity(total_mass, a_typical):
    """
    Approximate critical velocity for an encounter:
    when kinetic energy ≈ binding energy.
    """
    vcrit = np.sqrt(constants.G * total_mass / a_typical)
    return vcrit


# ======================================================
# Updated add_relative_motion for 6-body encounters
# ======================================================
def add_relative_motion(centers, v_coms, b, v_inf, phi, theta, psi):
    """
    Add impact parameter and velocity offset to the second triple (stars 4,5,6).
    Rotates relative vectors according to sampled Euler angles.
    """
    offset = np.array([20.0, b.value_in(units.AU), 0.0])  # AU offset
    vel = np.array([-v_inf.value_in(units.kms), 0.0, 0.0])  # approach along -x

    # Rotation matrices
    Rz1 = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi),  np.cos(phi), 0],
                    [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Rz2 = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0, 0, 1]])
    R = Rz2 @ Rx @ Rz1
    offset_rot = R @ offset
    vel_rot = R @ vel

    # First triple (stars 1,2,3) stays at origin
    # Second triple (stars 4,5,6) shifted
    for i in range(3,6):
        centers[i] = (np.array(centers[i]) + offset_rot).tolist()
        v_coms[i]  = (np.array(v_coms[i]) + vel_rot).tolist()

    return centers, v_coms


# ======================================================
# Main driver modifications
# ======================================================
# Initialize centers and COM velocities for six stars
# Initialize centers and COM velocities for 6 stars
centers = [[0.0, 0.0, 0.0] for _ in range(6)]
v_coms  = [[0.0, 0.0, 0.0] for _ in range(6)]




def classify_outcome(frames):
    """
    Classify the outcome of a 6-body triple–triple encounter.

    Returns integer state code:
      0 = Flyby (both triples remain bound)
      1 = Exchange (some stars swapped but bound systems remain)
      2 = Ionization (no bound pairs remain)
      3 = Merger (one or more physical mergers occurred)
      4 = Massive ejection (most massive star is unbound from all others)
      9 = Failed / undefined
    """
    from itertools import combinations
    import numpy as np
    from amuse.lab import constants, units

    try:
        final = frames[-1]
        N = len(final)
        if N == 0:
            return 9

        # --- Detect mergers ---
        m_values = final.mass.value_in(units.MSun)
        if N < 6 or np.max(m_values) > 1.3 * np.median(m_values):
            return 3  # merger detected

        # --- Check if the most massive star is unbound ---
        i_max = np.argmax(m_values)
        star_max = final[i_max]
        bound_to_any = False
        for j in range(N):
            if j == i_max:
                continue
            pj = final[j]
            r = (star_max.position - pj.position).length()
            v_rel = (star_max.velocity - pj.velocity).length()
            mu = (star_max.mass * pj.mass) / (star_max.mass + pj.mass)
            E_kin = 0.5 * mu * v_rel**2
            E_pot = -constants.G * star_max.mass * pj.mass / r
            if (E_kin + E_pot) < (0 | units.J):
                bound_to_any = True
                break
        if not bound_to_any:
            return 4  # massive ejection

        # --- Compute all pairwise binding energies ---
        bound_pairs = []
        for i, j in combinations(range(N), 2):
            pi, pj = final[i], final[j]
            r = (pi.position - pj.position).length()
            v_rel = (pi.velocity - pj.velocity).length()
            mu = (pi.mass * pj.mass) / (pi.mass + pj.mass)
            E_kin = 0.5 * mu * v_rel**2
            E_pot = -constants.G * pi.mass * pj.mass / r
            if (E_kin + E_pot) < (0 | units.J):
                bound_pairs.append((i, j))

        if len(bound_pairs) == 0:
            return 2  # ionization: all unbound

        # --- Build connectivity graph (union–find) ---
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for (i, j) in bound_pairs:
            union(i, j)

        n_clusters = len(set(find(k) for k in range(N)))

        if n_clusters <= 2:
            return 0  # flyby
        elif n_clusters <= 4:
            return 1  # exchange
        else:
            return 2  # ionization

    except Exception as e:
        print(f"[classify_outcome] failed: {e}")
        return 9


# ======================================================
# Main driver
# ======================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Monte Carlo triple–triple cross-section experiment")
    parser.add_argument("--velocity", type=float, default=1.0, help="Velocity in units of critical velocity")
    parser.add_argument("--n_b", type=int, default=20, help="Number of impact parameters to simulate")
    parser.add_argument("--sim_per_b", type=int, default=10, help="Number of simulations per impact parameter")
    parser.add_argument("--output", type=str, default="output_triple_cross", help="Output directory for results")
    args = parser.parse_args()

    v_factor = args.velocity
    n_b = args.n_b
    n_sim = args.sim_per_b
    save_path = args.output

    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in {save_path}")

    start_time = time.time()

    # Maximum impact parameter (rough estimate)
    b_max = 50 | units.AU
    impact_parameters = np.sqrt(np.linspace(0, b_max.value_in(units.AU)**2, n_b)) | units.AU

    # Prepare results array
    dtype = [
        ("v_inf", "f8"),
        ("b", "f8"),
        ("phi", "f8"),
        ("theta", "f8"),
        ("psi", "f8"),
        ("state", "u1"),
        ("max_mass", "f8"),
        ("max_velocity", "f8"),
        ("index", "u4"),
    ]
    results = np.zeros((len(impact_parameters) * n_sim,), dtype=dtype)

    index = 0
    for b in impact_parameters:
        print(f"--- Impact parameter b={b.in_(units.AU)} ---")

        phis = np.random.uniform(0, 2 * np.pi, n_sim)
        thetas = np.arccos(np.random.uniform(0, 1, n_sim))
        psis = np.random.uniform(0, 2 * np.pi, n_sim)

        for phi, theta, psi in zip(phis, thetas, psis):
            # Sample full initial conditions for this trial
            masses, sep, ecc, directions = sample_initial_conditions()

            # Compute typical binding scale and critical velocity for this set
            a_typical = np.mean(sep) | units.AU
            total_mass = sum(masses) | units.MSun
            vcrit = critical_velocity(total_mass, a_typical)
            v_inf = v_factor * vcrit

            # Initialize centers and COM velocities for 6 stars
            centers = [[0.0, 0.0, 0.0] for _ in range(6)]
            v_coms  = [[0.0, 0.0, 0.0] for _ in range(6)]

            # Apply relative motion for the second triple
            centers, v_coms = add_relative_motion(centers, v_coms, b, v_inf, phi, theta, psi)

            run_label = f"v{v_factor:.1f}_b{b.value_in(units.AU):.1f}_{index}"
            age = 3.5
            try:
                frames, max_mass, max_vel = run_6_body_simulation(
                    age,
                    masses,
                    sep,
                    ecc,
                    directions,
                    centers,
                    v_coms,
                    run_label=run_label
                )

                state = classify_outcome(frames)
                results[index] = (
                    v_inf.value_in(units.kms),
                    b.value_in(units.AU),
                    phi,
                    theta,
                    psi,
                    state,
                    max_mass.value_in(units.MSun),
                    max_vel.value_in(units.kms),
                    index,
                )

            except Exception as e:
                print(f"Simulation {index} failed: {e}")
                results[index] = (v_inf.value_in(units.kms), b.value_in(units.AU), phi, theta, psi, 9, 0, 0, index)

            index += 1

    np.save(os.path.join(save_path, f"results_velocity_{v_factor}.npy"), results)
    print(f"\nSimulation batch complete in {(time.time() - start_time)/60:.1f} min")
    print(f"Saved results to {save_path}")
