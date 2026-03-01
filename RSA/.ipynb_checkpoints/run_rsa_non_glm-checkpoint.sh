#!/usr/bin/env bash
#SBATCH --output=logs/%j-rsa_non_glm.out
#SBATCH -p psych_week
#SBATCH -t 50:00:00
#SBATCH --mem 100GB
#SBATCH -n 1

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/aa2842/conda_envs/myenv"
python "/gpfs/milgram/project/turk-browne/aa2842/multisensory-memory-project/RSA/rsa_nonglm.py"