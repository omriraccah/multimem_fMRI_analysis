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
module load dSQ

# Source conda directly instead of using conda init
source /gpfs/milgram/apps/avx2/software/miniconda/24.11.3/etc/profile.d/conda.sh

conda activate "/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem"

# Get the python path from the conda environment
PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"

# Run dSQBatch.py from the dSQ module's PATH (the old hardcoded
# /gpfs/.../hpc.rhel7/... path no longer exists after the OS migration).
dSQBatch.py --job-file /home/or62/project/multisensory-memory-project/RSA/joblist_ROI_Split_Half.txt --status-dir ./logs