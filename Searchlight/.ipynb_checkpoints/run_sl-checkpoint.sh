#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --array=0-40
#SBATCH --output=logs/searchlight-%j.out
#SBATCH --job-name searchlight
#SBATCH -p psych_week 
#SBATCH --mem-per-cpu 16G -t 100:00:00 --mail-type ALL -n 12 -c 1 -N 1
# Set up the environment
module load miniconda
module load OpenMPI
conda activate "/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem"

# Get the python path from the conda environment
PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"

# Run the python script with full path to Python (DO NOT EDIT LINE BELOW)
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /home/or62/project/multisensory-memory-project/Searchlight/joblist_searchlight_RSA.txt --status-dir ./logs