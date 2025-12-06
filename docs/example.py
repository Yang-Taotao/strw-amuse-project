"""
Example script from AMUSE lecture 2025-10-29.
"""

# imports
import numpy as np
from amuse.couple import bridge
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.protodisk import ProtoPlanetaryDisk
from amuse.lab import *
from matplotlib import pyplot as plt

# here is my lirbary


def plot_solar_system_with_disk(sosy, disk, model_time):
    m = 100 * np.sqrt(sosy.mass.value_in(units.MJupiter))
    m[0] = 100
    figure = plt.figure(figsize=(10, 10))
    plt.title(f"t={model_time.in_(units.yr)}")
    earth = sosy[sosy.name == "EARTHMOO"]
    plt.scatter(sosy.x.value_in(units.au), sosy.y.value_in(units.au), c="r", s=m)
    plt.scatter(disk.x.value_in(units.au), disk.y.value_in(units.au), c="k", s=10)
    plt.scatter(earth.x.value_in(units.au), earth.y.value_in(units.au), c="b", s=50)
    plt.show()


if __name__ in ("__main__"):

    # here is my executable script

    sosy = new_solar_system()
    converter = nbody_system.nbody_to_si(sosy.mass.sum(), 1 | units.au)

    gravity = Hermite(converter)
    gravity.particles.add_particles(sosy)
    ch2_sosy = gravity.particles.new_channel_to(sosy)

    saturn = sosy[sosy.name == "SATURN"]
    sun = sosy[sosy.name == "SUN"]
    orbit = orbital_elements_from_binary(sun + saturn)
    a_saturn = orbit[2]
    RHill = (a_saturn * (saturn.mass / sun.mass) ** (1 / 3))[0]
    Ndisk = 1000
    converter = nbody_system.nbody_to_si(saturn.mass.sum(), RHill)

    disk = ProtoPlanetaryDisk(
        Ndisk, convert_nbody=converter, Rmin=0.1, Rmax=1, q_out=1.0, discfraction=1000
    ).result
    disk.position += saturn.position
    disk.velocity += saturn.velocity
    model_time = 0 | units.yr
    plot_solar_system_with_disk(sosy, disk, model_time)

    hydro = Fi(converter, mode="openmp")
    hydro.parameters.timestep = 0.001 | units.yr
    hydro.particles.add_particles(disk)
    ch2_disk = hydro.particles.new_channel_to(disk)

    gravhydro = bridge.Bridge()
    gravhydro.add_system(gravity, (hydro,))
    gravhydro.add_system(hydro, (gravity,))
    dt = 1 | units.yr
    gravhydro.timestep = 0.01 * dt

    E0 = gravity.kinetic_energy + gravity.potential_energy
    t_end = 100 | units.yr
    while model_time < t_end:
        model_time += dt
        gravhydro.evolve_model(model_time)
        ch2_disk.copy()
        ch2_sosy.copy()

        E = gravity.kinetic_energy + gravity.potential_energy
        print(f"t={model_time.in_(units.yr)}, dE/E = {(E-E0)/E0}")

        plot_solar_system_with_disk(sosy, disk, model_time)

    gravity.stop()
    hydro.stop()
