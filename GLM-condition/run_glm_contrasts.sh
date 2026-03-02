#!/usr/bin/env bash
#SBATCH --output=logs/%j-glm_contrasts.out
#SBATCH -p psych_day
#SBATCH -t 5:00:00
#SBATCH --mem 50GB
#SBATCH -n 1

module load miniconda
conda activate "./myenv"
python "./GLM-condition/glm_contrasts.py"