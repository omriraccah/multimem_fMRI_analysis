#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --array=0-29
#SBATCH --output=logs/%j-rsa.out
#SBATCH --job-name RSA
#SBATCH -p psych_week 
#SBATCH --mem-per-cpu 25G -t 100:00:00 --mail-type ALL -n 1 -c 1 -N 1

# Set up the environment
module load miniconda
module load OpenMPI

# Source conda directly instead of using conda init
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh

conda activate "./myenv_multimem"

# Get the python path from the conda environment
PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"

# Run the python script with full path to Python (DO NOT EDIT LINE BELOW)
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file ./RSA/joblist_ROI_RSA.txt --status-dir ./logs