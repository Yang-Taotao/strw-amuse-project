import numpy as np
from amuse.units import units
from run_6body_encounter import run_6_body_simulation
from helpers import outcomes as classify_outcome
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def sample_initial_conditions(N_binaries=3):
    """Sample initial conditions for N_binaries colliding binaries."""
    f = np.random.uniform(0, 2*np.pi, N_binaries)
    sep = [float(1.0 / (2.0 * np.cos(fi/2.0)**2)) for fi in f]
    ecc = [float(e) for e in np.random.uniform(0.0, 0.9, N_binaries)]
    phi = np.random.uniform(0, 2*np.pi, N_binaries)
    theta = np.random.uniform(0, np.pi/2, N_binaries)
    psi = np.random.uniform(0, 2*np.pi, N_binaries)

    directions = []
    for i in range(N_binaries):
        vec = np.array([np.cos(phi[i])*np.sin(theta[i]), 0, np.sin(phi[i])*np.sin(theta[i])])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            vec = np.array([1.0,0.0,0.0])
        directions.append(float(vec[0]))  # keep scalar if run_6_body_simulation expects scalar

    centers = []
    for i in range(N_binaries):
        c = np.array([20.0*np.cos(psi[i]), 20.0*np.sin(psi[i]), 0.0])
        centers.append(c.tolist())

    b = [float(bi) for bi in np.random.uniform(0.0, 20.0, N_binaries)]
    v_coms = [[float(np.random.uniform(0,50.0)), 0.0, 0.0] for _ in range(N_binaries)]

    return sep, ecc, directions, centers, b, v_coms

def sequential_top_creative_collision(N_trials=100, N_best=5, N_binaries=3, distance=100.0):
    """Run N_trials and keep top N_best creative_collision cases."""
    
    if isinstance(distance, (float,int)):
        distance = [distance, distance]
    
    top_trials = []  # stores dicts with parameters + outcome info

    print_every = max(1, N_trials // 10)

    for idx in range(N_trials):
        
        print(f"Simulation {idx+1}/{N_trials}... ", end='')

        sep, ecc, directions, centers, b, v_coms = sample_initial_conditions(N_binaries=N_binaries)
        
        try:
            frames, max_mass, max_velocity, outcome = run_6_body_simulation(
                sep, ecc, directions, v_coms,
                centers, b, distance=distance,
                masses=None, centers=None, age=None,
                run_label=f"Trial_{idx+1}"
            )
        except Exception as e:
            print(f"Simulation {idx+1} failed: {e}")
            continue

        # Determine outcome label and description
        if isinstance(outcome, str):
            label = outcome
            description = f"Pre-classified: {outcome}"
        else:
            try:
                label, description = classify_outcome(
                    initial_particles=frames[0],
                    final_particles=frames[-1],
                    max_mass=max_mass
                )
            except Exception as e:
                label = 'unclassified'
                description = f"Error: {e}"

        # If creative collision, record trial
        if label in ['creative_ionized']:
            trial_info = {
                'sep': sep,
                'ecc': ecc,
                'directions': directions,
                'centers': centers,
                'b': b,
                'v_coms': v_coms,
                'max_mass': max_mass,
                'max_velocity': max_velocity,
                'label': label,
                'description': description
            }
            top_trials.append(trial_info)
            # Sort by most massive star's velocity for creative_ionized, largest first
            top_trials = sorted(top_trials, key=lambda x: x['max_velocity'].value_in(units.km/units.s), reverse=True)
            top_trials = top_trials[:N_best]  # keep only top N_best

        if (idx % print_every) == 0:
            v_kms = max_velocity.value_in(units.km/units.s)
            print(f"Most massive star velocity: {v_kms:.2f} km/s, outcome: {label}")

    print(f"\nCollected top {len(top_trials)} creative collision trials.")
    return top_trials
