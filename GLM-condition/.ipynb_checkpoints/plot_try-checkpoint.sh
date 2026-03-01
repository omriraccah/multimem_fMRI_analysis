#!/usr/bin/env bash
#SBATCH --output=logs/%j-SURFPLOT.out
#SBATCH -p psych_day
#SBATCH -t 5:00:00
#SBATCH --mem 100GB
#SBATCH -n 12

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem"

python "/gpfs/milgram/project/turk-browne/or62/multisensory-memory-project/GLM-condition/plot_try.py" 