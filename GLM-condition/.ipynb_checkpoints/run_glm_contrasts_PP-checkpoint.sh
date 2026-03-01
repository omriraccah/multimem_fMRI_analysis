#!/usr/bin/env bash
#SBATCH --output logs/%j-GLM-contrasts_%A_%a.out
#SBATCH --job-name GLM_CONTRASTS-PP
#SBATCH --array=1-30
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=30:00
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day

module load miniconda
conda activate "/gpfs/milgram/project/turk-browne/$1/conda_envs/myenv"

SUBJECTS=("01" "02" "03" "05" "06" "07" "08" "09" "10" "11" \
          "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" \
          "23" "24" "25" "26" "27" "28" "29" "30" "31" "32")

SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID - 1]}

echo "Running GLM contrasts for subject $SUBJECT_ID"

python "/gpfs/milgram/project/turk-browne/$1/multisensory-memory-project/GLM-condition/glm_contrasts.py" -s "$SUBJECT_ID"