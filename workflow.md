# Main

## Revised idea <- Important

### Notes from consulting with Simon

A rough outline of the revised idea:

- General: shoot triple/bin at some triple/bin sys
- Localized environment of space
- Fixed host triplet/quintuplet system <- shoot some bin/triplet at the host from all points on a sphere
- How many of these trails have pistol ejected -> prob
- More likely to form in bin-bin-bin or tri-tri
- Assume close encounter event of bin-bin-bin or tri-tri has happened in some cluster
- Distant effect of other stars in the cluster -> trivial

### In detail

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

### Additional

> Snapshots -> plt

## passing grade req

temp

## Legacy idea

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
    if close_encouter:
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
def psudo_prob(num_true_flag, num_config): #????? -> MC? -> Literature
    num_flag = num_true_flag
    return num_flag/num_config # prob of pistol inside selected number of configs
```

```py
def plotter(args):
    *args = mass, location, time
    plt.plot(args) # cmap with numbered output
    plt.save(f"cmap_{time}")
```
