"""
Main script for running the 6-body encounter simulation and visualization.
"""

# %%
# import
from src.strw_amuse.config import PARAM_REF, PARAM_TEST
from src.strw_amuse.run_simulation import run_6_body_simulation
from src.strw_amuse.gif_plotting import visualize_frames, visualize_initial_final_frames

# %%
# Run sim - no encounter
frames_ref, outcome_ref = run_6_body_simulation(*PARAM_REF)

# %%
# Visualize - no encounter -> no collisions
visualize_initial_final_frames(frames=frames_ref, run_label="ref_case")
visualize_frames(frames=frames_ref, run_label="ref_case")

# %%
# Run sim - ionized
frames_test, outcome_test = run_6_body_simulation(*PARAM_TEST)

# %%
# Visualize - collisions -> creative ionized
visualize_initial_final_frames(frames=frames_test, run_label="test_case")
visualize_frames(frames=frames_test, run_label="test_case")
