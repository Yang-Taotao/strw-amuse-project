"""
Main script for running the 6-body encounter simulation and visualization.
"""

# import
from src.strw_amuse.run_6body_encounter import run_6_body_simulation
from src.strw_amuse.gif_plotting import visualize_frames

# ------------------------------ #
# No encounter (ref case)
# ------------------------------ #
# Set initial conditions
age = 3.5  # Myr
masses = [90, 10, 20, 70, 10, 10]  # Msun
sep = [30, 20, 50]  # AU
ecc = [0.0, 0.0, 0]
centers = [
    [-100, 0, 0],
    [300, 0, 0],
    [0, 600, 0],
]  # AU
v_coms = [
    [30.0, 3.0, 0.0],
    [-10.0, -21.0, 0.0],
    [5.0, -20.0, 1.0],
]  # km/s
direction = [0.4, -0.6, 1.2]

# Run sim
no_colission, _, _ = run_6_body_simulation(
    age=age,
    masses=masses,
    sep=sep,
    ecc=ecc,
    direction=direction,
    centers=centers,
    v_coms=v_coms,
    run_label="Test_No_Collision",
)
# Visualize
visualize_frames(frames=no_colission, run_label="Test_No collision")

# ------------------------------ #
# Encounter with collision
# ------------------------------ #
# Set initial conditions
age = 3.5
masses = [80, 40, 30, 20, 40, 10]
sep = [0.5, 1.0, 1.5]
ecc = [0.6, 0.2, 0.1]
centers = [
    [0.0, 0.0, 0.0],
    [100.0, 50.0, 0.0],
    [-300.0, 50.0, 0.0],
]
v_coms = [
    [0.05, 0.00, 0.00],
    [0.00, -0.05, 0.00],
    [0.00, 0.05, 0.00],
]
directions = [0.0, 0.4, -0.4]

# Run sim
frames, max_mass, max_vel = run_6_body_simulation(
    age=age,
    masses=masses,
    sep=sep,
    ecc=ecc,
    direction=directions,
    centers=centers,
    v_coms=v_coms,
    run_label="Run_for_ejection",
)

# Visualize
visualize_frames(frames, "Collision")
