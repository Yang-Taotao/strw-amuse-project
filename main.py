"""
Main script for running the 6-body encounter simulation and visualization.
"""

# %%
# import
import numpy as np

from src.strw_amuse.run_simulation import run_6_body_simulation
from src.strw_amuse.gif_plotting import visualize_frames, visualize_initial_final_frames

# %%
# No encounter (ref case)
# set initial config for test param
sep = [30, 20, 10]  # AU
ecc = [0.0, 0.0, 0]
v_mag = [2, 1]
impact_parameter = [0.0, 10.0]  # AU
theta = [np.pi / 3, np.pi / 4]
phi = [np.pi, np.pi / 2]
psi = [np.pi / 2, np.pi]
distance = [100, 100]  # AU
true_anomalies = [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4]
run_label = "test_no_collision"
# build param
param_no_collision = (
    sep,  # 3
    true_anomalies,  # 3
    ecc,  # 3
    theta,  # 2
    phi,  # 2
    v_mag,  # 2
    impact_parameter,  # 2
    psi,  # 2
    distance,
    run_label,
)
# %%
# Run sim - no encounter
frames_no_collision, outcome = run_6_body_simulation(*param_no_collision)

# %%
# Visualize - no encounter
visualize_initial_final_frames(frames_no_collision, run_label="No_Collision_Outcome")
visualize_frames(frames=frames_no_collision, run_label="Test_Collision_False")

# %%
# Encounter with collision
# set initial conditions for incoming bin B & C
sep = [30, 20, 10]  # AU
ecc = [0.0, 0.0, 0]
v_mag = [0.2, 0.2]
impact_parameter = [0.0, 0.0]  # AU
theta = [np.pi / 4, np.pi / 4]
phi = [np.pi / 2, np.pi / 2]
psi = [np.pi / 2, np.pi / 2]

distance = [50, 50]  # AU
true_anomalies = [3 * np.pi / 4, np.pi / 4, 7 * np.pi / 4]
run_label = "Creative_collision_ionized"
# build param
param_collision_ionized = (
    sep,  # 3
    true_anomalies,  # 3
    ecc,  # 3
    theta,  # 2
    phi,  # 2
    v_mag,  # 2
    impact_parameter,  # 2
    psi,  # 2
    distance,
    run_label,
)
# %%
# Run sim - no encounter
frames_collision_ionized, outcome = run_6_body_simulation(*param_collision_ionized)

# %%
# Visualize - collisions
visualize_initial_final_frames(
    frames_collision_ionized, run_label="Creative_Collision_Ionized_Outcome"
)
visualize_frames(frames_collision_ionized, run_label="Test_Creative_Collision_Ionized")
