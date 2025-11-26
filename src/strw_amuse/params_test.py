import numpy as np


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from amuse.units import units
from amuse.ext.orbital_elements import generate_binaries
from amuse.lab import *
from amuse.io import write_set_to_file, read_set_from_file

# import libraries
from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles

from itertools import combinations
import helpers
import run_simulation
np.random.seed(1)
m_min=10
m_max=50
vel_min=0
vel_max=1
position_min=-1000
position_max=1000
sep_min=1
sep_max=30
ecc_min=0
ecc_max=1
theta_min=0
theta_max=180
phi_min=0
phi_max=180
phase_min=0
phase_max=180
direc_min=-1
direc_max=1
imp_min=0
imp_max=100
plane_min=-1
plane_max=1
n_bin=3
n_trip=0
runs=2
psi_min=0
psi_max=180
separations,eccs,velocities,phis,anomalies,thetas,impact,masses,position,psis=helpers.vector_params(runs,
                                                                                           n_bin,
                                                                                             n_trip,
                                                                                               m_min,
                                                                                               m_max,
                                                                                                 vel_min,
                                                                                                 vel_max,
                                                                                                   position_min,
                                                                                                   position_max,
                                                                                                     sep_min,sep_max,
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
                                                                                                     psi_max)
print(velocities[0][0])
sim=run_simulation.run_6_body_simulation(
    separations[0],
    anomalies[0],
    eccs[0],
    thetas[0],
    phis[0],
    velocities[0],
    impact[0],
    psis[0],
    position[0],
    'bin',
    masses[0],
    centers=None,  # <-- impact orientation angles
    )