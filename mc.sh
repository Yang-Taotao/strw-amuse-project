#!/bin/bash
#SBATCH --job-name=pistol_mc_mp
#SBATCH --export=NONE
#SBATCH --output=mc_data/output_%A_%a.out
#SBATCH --error=mc_data/error_%A_%a.err
#SBATCH --mail-user="YOUR_EMAIL_HERE"
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-9

module purge
module load ALICE/default
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amuse-py313

export CWD=$(pwd)
export DATE=$(date)

echo "[$SHELL] #### Starting Script"
echo "[$SHELL] ## USER: $SLURM_JOB_USER | ID: $SLURM_JOB_ID | TASK"
echo "[$SHELL] ## current working directory: "$CWD
echo "[$SHELL] ## Run script"

python mc_pool_hpc.py

echo "[$SHELL] ## Script finished"
echo "[$SHELL] ## Job done "$DATE
echo "[$SHELL] ## Used $SLURM_NTASKS cores"
echo "[$SHELL] ## Used $SLURM_CPUS_ON_NODE processors/CPUS"
echo "[$SHELL] ## Used $SLURM_CPUS_PER_TASK processors/CPUS per task"
echo "[$SHELL] #### Finished Monte Carlo. Task complete."