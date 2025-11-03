---
Author: Taotao Yang
Date Created: 2025-10-02
---
# Installation gudie for AMUSE v2025.9.0

In this guide, we will give a step-by-step process of how to properly install AMUSE on you Windows computer.

## Initial setup

- Install WSL on your device, the default Ubuntu distribution is good.
- Have your git details sorted.
- Have some sort of IDE/Editor installed.
- Optionally, you should have a terminal/shell with some font that you prefer also installed for QoL upgrades.
- Use you linux file system for file storage, it is much easier to manage them from within Linux

## Conda Installation

Start you terminal up, and get into your home directory of Linux file system with:

```bash
> cd ~
```

Let's do a quick update:

```bash
> sudo apt update && sudo apt upgrade
```

We should get conda sorted:

```bash
> curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Upon downloading, you can install the miniforge version of conda with:

```bash
> bash Miniforge3-$(uname)-$(uname -m).sh
```

We should now reload our shell to get conda working:

```bash
> source ~/.bashrc
> conda init
> exec bash
```

We can see that the correct version of conda is installed with:

```bash
> conda --version
```

Now, in your home directory, you could see a list of your files with:

```bash
> ls -a
```

We can now create a simple folder to put all our projects in. Say that our folder name is simply 'repo':

```bash
> mkdir repo
```

We can now enter the folder with:

```bash
> cd ~/repo
```

## Creating conda environment

We should now create an environment with conda to contain our future project regarding AMUSE:

```bash
> conda create -n amuse-py313 python=3.13
```

We should now activate our conda env:

```bash
> conda activate amuse-py313
```

Note: you should always work under a peoject dependent environment for managing your packages and installations.

## AMUSE Installation

At this stage, we move on to install AMUSE, we will clone it from git:

```bash
> cd ~/repo
> git clone https://github.com/amusecode/amuse.git
```

We now switch to the desired release branch and install:

```bash
> git checkout v2025.9.0
> ./setup
```

Now, following the prompt, you will see that some dependencies may be needed, simply follow along to install them. Afterwards, you would finish the installation with:

```bash
> ./setup install amuse-framework
```

You can additionally run a series of automated tests to see that the installed amuse-framework module is functional:

```bash
> ./setup test amuse-framework
```

We should fix the MPI error on Ubuntu with:

```bash
> echo 'btl_tcp_if_include=lo' >>.openmpi/mca-params.conf
```

If error shows up indicating the file does not exist, simply go to `~/.openmpi` and create `mca-params.conf` with `btl_tcp_if_include=lo` as the sole content.

It is also great to just install everything amuse has to offer with:

```bash
> ./setup install all
```
