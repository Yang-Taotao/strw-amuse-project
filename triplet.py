import numpy as np
import itertools
import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from amuse.units import units

from amuse.lab import *
from amuse.io import write_set_to_file, read_set_from_file

#import libraries
from amuse.lab import *
from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles

from itertools import combinations




def make_trinary_system(
    masses,
    positions,
    velocities
):
        "Function takes list of masses, must be size 3, also takes a 3x3 array of positions and velocities"
        "Stars are initialized to the position of the star at position 0, so take this into account"

        stars=Particles(3)
        for i in range(3):
            stars[i].mass=masses[i]|units.Msun
            stars[i].velocity=VectorQuantity(velocities[i],units.kms)
            stars[i].name=f'T{i}'
            stars[i].position=VectorQuantity(positions[i],units.au)
            if i!=0:
                stars[i].position+=stars[0].position
                stars[i].velocity+=stars[0].velocity
        return stars
    