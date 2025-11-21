"""
Main script for running the 6-body encounter simulation and visualization.
"""

# %%
# import
from src.strw_amuse.run_simulation import run_6_body_simulation
from src.strw_amuse.gif_plotting import visualize_frames

# %%
# No encounter (ref case)
# set initial config for test param
sep = [30, 20, 50]  # AU
ecc = [0.0, 0.0, 0]
direction = [0.4, -0.6, 1.2]
v_coms = [
    [30.0, 3.0, 0.0],
    [-10.0, -21.0, 0.0],
    [5.0, -20.0, 1.0],
]  # km/s
orbit_plane = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
impact_parameter = [0.0, 0.0]  # AU
distance = [-100, 100]  # AU
masses = [90, 10, 20, 70, 10, 10]  # Msun
centers = [
    [-100, 0, 0],
    [300, 0, 0],
    [0, 600, 0],
]  # AU
age = 3.5  # Myr
run_label = "test_no_collision"
# build param
param_no_collision = (
    sep,
    ecc,
    direction,
    v_coms,
    orbit_plane,
    impact_parameter,
    distance,
    masses,
    centers,
    age,
    run_label,
)
# %%
# Run sim - no encounter
frames_no_collision = run_6_body_simulation(*param_no_collision)

# %%
# Visualize - no encounter
visualize_frames(frames=frames_no_collision, run_label="Test_Collision_False")

# %%
# Encounter with collision
# set initial conditions
sep = [0.5, 1.0, 1.5]
ecc = [0.6, 0.2, 0.1]
directions = [0.0, 0.4, -0.4]
v_coms = [
    [0.05, 0.00, 0.00],
    [0.00, -0.05, 0.00],
    [0.00, 0.05, 0.00],
]
orbit_plane = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
impact_parameter = [0.0, 0.0]  # AU
distance = [-1, 1]  # AU
masses = [80, 40, 30, 20, 40, 10]
centers = [
    [0.0, 0.0, 0.0],
    [100.0, 50.0, 0.0],
    [-300.0, 50.0, 0.0],
]
age = 3.5  # Myr
run_label = "test_with_collision"
# build param
param_with_collision = (
    sep,
    ecc,
    direction,
    v_coms,
    orbit_plane,
    impact_parameter,
    distance,
    masses,
    centers,
    age,
    run_label,
)
# %%
# Run sim - collisions
frames_collision = run_6_body_simulation(*param_with_collision)

# %%
# Visualize - collisions
visualize_frames(frames_collision, "Test_Collision_True")
