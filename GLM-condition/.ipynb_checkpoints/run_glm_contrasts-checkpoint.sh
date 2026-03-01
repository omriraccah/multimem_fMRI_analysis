#!/usr/bin/env bash
#SBATCH --output=logs/%j-glm_contrasts.out
#SBATCH -p psych_day
#SBATCH -t 5:00:00
#SBATCH --mem 50GB
#SBATCH -n 1

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$1/conda_envs/myenv"
python "/gpfs/milgram/project/turk-browne/$1/multisensory-memory-project/GLM-condition/glm_contrasts.py"