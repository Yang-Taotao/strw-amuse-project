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
    - [Assumptions](#assumptions)
    - [Some interesting questions](#some-interesting-questions)
  - [Example](#example)
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

- Run project with [`main.py`](./main.py)
- Check [`Documentation`](./docs/)(./docs/AMUSE_Install_v2025.9.0.md) for `AMUSE` installation setup and example script for `AMUSE` simulations.
- Require `AMUSE` [`v2025.9.0`](https://github.com/amusecode/amuse/releases/tag/v2025.9.0) for current project.
- See [`environment.yml`](./environment.yml) for `Conda` environment.
- See [`pyproject.toml`](./pyproject.toml) for `Black` config.
- See [`setup.cfg`](./setup.cfg) for `isort` and `flake8` config.

## Goals

Generate probability estimation of pistol star formation based off of some configurations of star system interactions inside a cluster.

### Idea

- Shoot a triplet or binary at some host triplet or binary-binary system.
- Use host triplet/quintuplet system as point of reference.
- Get probability of these trails ejecting pistol star.
  
### Assumptions

- Close encounter event of bin-bin-bin or tri-tri has happened in some cluster.
- Interaction take place in localized spatial environment
- Distant effect of other stars in the cluster is trivial.

### Some interesting questions

- Is it more likely to form in bin-bin-bin or tri-tri configuration?

## Example

Use the following example script as provided in [`main.py`](main.py) as an example on how to run one simulation run of this project.

```py
import src.strw_amuse as project

project.utils.logger.setup_logging()
project.utils.checker.check_sim_example()

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

</details>
