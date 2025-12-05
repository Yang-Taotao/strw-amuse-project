# strw-amuse-project

This is the README document for strw-amuse-project, a project for the 2025 [***Simulation and Modeling in Astrophysics (AMUSE)***](https://studiegids.universiteitleiden.nl/en/courses/130588/simulation-and-modeling-in-astrophysics-amuse) lecture hosted at [**Leiden Observatory**](https://local.strw.leidenuniv.nl/).

**Dated:** 2025-October-04

## Index

- [strw-amuse-project](#strw-amuse-project)
  - [Index](#index)
  - [Collaborators](#collaborators)
  - [Usage and Notes](#usage-and-notes)
  - [Goals](#goals)
    - [Idea](#idea)
    - [Example I/O](#example-io)
    - [Pipeline](#pipeline)
  - [Grading rubric](#grading-rubric)
  - [Archived](#archived)
    - [Steps](#steps)
    - [Configs](#configs)
    - [Alternative Approach](#alternative-approach)

## Collaborators

We list the collaborators in alphabetical order.

| Collaborator           | Contact                                                                 | Program       | Institution       |
| ---------------------- |------------------------------------------------------------------------ | ------------- | ----------------- |
| **Eirini Chrysovergi** | [chrysovergi@strw.leidenuniv.nl](mailto:chrysovergi@strw.leidenuniv.nl) | MSc Astronomy | Leiden University |
| **Marc Seegers**       | [seegers@strw.leidenuniv.nl](mailto:seegers@strw.leidenuniv.nl)         | MSc Astronomy | Leiden University |
| **Taotao Yang**        | [tyang@strw.leidenuniv.nl](mailto:tyang@strw.leidenuniv.nl)             | MSc Astronomy | Leiden University |

## Usage and Notes

![Linux](https://img.shields.io/badge/-Linux-grey?logo=linux)

- Run scripts from [`main.py`](./main.py)
- Check [`Documentation`](./docs/AMUSE_Install_v2025.9.0.md) for AMUSE installation setup and example script for simulations.
- AMUSE [`v2025.9.0`](https://github.com/amusecode/amuse/releases/tag/v2025.9.0) required for current project.
- Conda [`env`](./environment.yml) recommended. Replicate with provided file.
- See [`pyproject.toml`](./pyproject.toml) for `Black` config.
- See [`setup.cfg`](./setup.cfg) for `isort` and `flake8` config.

## Goals

Generate probability estimation of pistol star formation based off of some configurations of star system interactions in a cluster.

### Idea

A rough outline follows:

- General: shoot a triplet or binary at some host triplet and binary-binary system
- Localized space environment
- Fixed host triplet/quintuplet system <- shoot some bin/triplet at the host from all points on a sphere
- How many of these trails have pistol ejected -> probability
- More likely to form in bin-bin-bin or tri-tri configuration
- Assume close encounter event of bin-bin-bin or tri-tri has happened in some cluster
- Distant effect of other stars in the cluster -> trivial

### Example I/O

> I/O
>
> Input:
>
> - host_system_param(triplet or bin-bin) <- mass, velocity, location, separation, etc
> - variable_system_param(triplet or bin) <- mass, velocity, location, separation, etc
>
> Output:\
>
> - ejection_flag(True or False)
> - ejected_param(mass, velocity)
>

### Pipeline

> I/O
>
> 1. Set CONFIG file or entries -> config.py
> 2. Build parameter_array enteries based off of CONFIG -> param = param_build(config)
> 3. Run simulation with parameter_array -> sim_results = run_sim(param)
> 4. Visualization and probability estimation -> output = visualization(sim_results)
>

Ideally, we should be able to call and use the package as follows:

```py
# import
from project_name import config, magic
# generate results
magic(config)
```

## Grading rubric

***Section WIP***

---

## Archived

<details>
<summary>Read More...</summary>

### Steps

> I/O:
>
>input param -> config (init conditions)\
>func()\
>output -> Ejected (T/F) -> if ejected -> ejected item (mass, velocity)\
> check if ejected item is max mass item in system -> if not -> continue till max mass ejected

This summarize into:

> Overall:
>
> a whole wow of input param <- find optimal params to use <- MC?\
> run all configs\
> get all results\
> output -> num_max_mass_ejected/total_configs <- prob of max mass ejection\
> output -> the max mass of the ejected max mass items

### Configs

> param:
>
> mass\
> eccentricity <- if bin\
> separation <- if bin\
> centers of mass/bin\
> velocity and directions

```py
def param_builder(some_param):
    *some_param = (
        config,
        if_bin,
        num_bin,
        num_single,
        masses,
        velocity,
        separation,
        centers,
        direction,
    )
    if config == (4, 2): # 4 bin 2 single
        param = (masses of bin, mass of single, centers, direction, velocity, separation)
    elif config == (0, 6):
        param = sss
    return param
```

```py
def func(some_param):
    param = param_builder(some_param)
    run_sim(param)
    results = (mass, velocity, spin)
    return results
```

### Alternative Approach

```py
init_param = (n_bin, n_single, r_cluster, age_cluster)
```

```py
def particle_assign(init_param):
    masses, sep, velocity, location = model(init_param)
    intermediate_param = (
        masses,
        sep,
        velocity,
        location,
    )
    return intermediate_param
```

```py
def sim(intermediate_param, time_step):

    # add
    stellar_evolution
    gravity
    
    # hydro
    if close_encounter:
        hydro()
        replace collide with new
    
    # ejection
    if ejected:
        # mass check
        if mass < max_mass:
            continue
        else:
            results = (
                (mass_ejected_star, velocity, spin),    # ejected results
                (mass_cluster, r_cluster),              # cluster results
                ejected_flag,                           # if ejected at this config
            )
    
    # plot
    plotter()
    # return
    return results
```

```py
def pseudo_prob(num_true_flag, num_config): #????? -> MC? -> Literature
    num_flag = num_true_flag
    return num_flag/num_config # prob of pistol inside selected number of configs
```

```py
def plotter(args):
    *args = mass, location, time
    plt.plot(args) # cmap with numbered output
    plt.save(f"cmap_{time}")
```

<\details>
