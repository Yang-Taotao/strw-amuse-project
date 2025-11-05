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


def make_binary(m1, m2, a, e=0.0, center=None, direction=0.0, orbit_plane=[0, 0, 1]):
    """
    Create a binary system with arbitrary eccentricity and orientation.

    Parameters
    ----------
    m1, m2 : float or Quantity
        Masses in Msun.
    a : Quantity
        Semi-major axis (AU).
    e : float
        Eccentricity (0=circular, 0<e<1=elliptical).
    center : VectorQuantity or list
        Center-of-mass position (default: [0,0,0] AU).
    direction : float
        Rotation angle around z-axis (radians) to orient the orbit.
    orbit_plane : list of 3 floats
        Normal vector defining orbital plane. Default: z-axis.

    Returns
    -------
    p1, p2 : Particle
        Two AMUSE particles with positions and velocities.
    """
    a=a|units.au
    m1 = m1 | units.MSun
    m2 = m2 | units.MSun
    total_mass = m1 + m2

    # Default center
    if center is None:
        center = VectorQuantity([0,0,0], units.AU)
    elif not isinstance(center, VectorQuantity):
        center = VectorQuantity(center, units.AU)

    # Circular approximation if e=0
    # More generally, sample at pericenter for simplicity
    r_rel = a * (1 - e)  # separation at pericenter
    r1 = -(m2 / total_mass) * r_rel
    r2 =  (m1 / total_mass) * r_rel

    # Rotation matrix around z (or orbit_plane)
    c, s = np.cos(direction), np.sin(direction)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    pos1 = np.dot(R, [r1.value_in(units.AU), 0., 0.]) | units.AU
    pos2 = np.dot(R, [r2.value_in(units.AU), 0., 0.]) | units.AU

    p1 = Particle(mass=m1)
    p2 = Particle(mass=m2)
    p1.position = center + pos1
    p2.position = center + pos2

    # Circular or elliptical orbit velocity
    if e == 0.0:
        # circular orbit
        v_rel = (constants.G * total_mass / a)**0.5
    elif e < 1.0:
        # elliptical
        v_rel = ((constants.G * total_mass * float(1 + e) / (a * float(1 - e)))**0.5)
    else:
        raise ValueError("Eccentricity cannot be > or = 1")


    v1 = + (m2 / total_mass) * v_rel
    v2 = - (m1 / total_mass) * v_rel

    vel1 = np.dot(R, [0., v1.value_in(units.kms), 0.]) | units.kms
    vel2 = np.dot(R, [0., v2.value_in(units.kms), 0.]) | units.kms
    p1.velocity = vel1
    p2.velocity = vel2

    return p1, p2

def N_systems(N_binaries,N_triplets,mas_bin=None,mas_triplet=None,a_bin=None,pos_triplet=None,pos_bin=None,vel_triplet=None):
    "This function makes N different binary or trinary systems, parameters are:"
    "N_binaries: int"
    "Number of binaries"
    "N_triplets: int"
    "Number of triplet systems"
    "mas_bin: N*2 array"
    "Mass of binaries (MSun)"
    "mas_triplets: N*3 array"
    "Mass of triplets (MSun)"
    "a_bin:list of size N"
    "Semimajor axis of binary partner"
    "pos_triplets: N*3*3 array"
    "Positions of trinaries (AU)"
    "pos_bin: N*3 array"
    "Center of mass position for binary"
    "vel_triplet: N*3*3 array"
    "Velocities of triplets (kms)"
    "Do not enter units in inputs, they get entered automatically"
    
    binaries=[]
    triplets=[]
    if N_binaries>0:
        
        for i in range(N_binaries):
            p1,p2=make_binary(mas_bin[i][0],mas_bin[i][1],a=a_bin[i],e=0.0,center=pos_bin[i])
            binaries.append([p1,p2])
    else:
        pass
    if N_triplets>0:
        for i in range(N_triplets):
            stars=make_trinary_system(mas_triplet[i],pos_triplet[i],vel_triplet[i])
            triplets.append(stars)
           
    else:
        pass
    if N_binaries==0 and N_triplets==0:
        return print('No binaries or triplets inserted')
    elif N_binaries!=0 and N_triplets==0:
        return binaries
    elif N_binaries==0 and N_triplets!=0:
        return triplets
    elif N_binaries!=0 and N_triplets!=0:
        return binaries,triplets
mas_trip=[[100,
          10,
          5]
         ]
pos_trip=[[[500,70,0],
           [20,10,0],
           [30,15,0]]]
vel_trip=[[[20,10,0],
           [5,2,0],
           [2,1,0]]]

stars=N_systems(0,1,mas_triplet=mas_trip,pos_triplet=pos_trip,vel_triplet=vel_trip)
print(stars)