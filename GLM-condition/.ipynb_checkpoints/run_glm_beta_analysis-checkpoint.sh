#!/usr/bin/env bash
#SBATCH --output=logs/%j-glm_beta_analysis.out
#SBATCH -p psych_day
#SBATCH -t 5:00:00
#SBATCH --mem 100GB
#SBATCH -n 1

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$1/conda_envs/myenv"
python "/gpfs/milgram/project/turk-browne/$1/multisensory-memory-project/GLM-condition/glm_beta_analysis.py"