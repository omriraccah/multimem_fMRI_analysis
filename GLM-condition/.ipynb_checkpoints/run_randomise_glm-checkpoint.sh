#!/bin/bash
#SBATCH --mem-per-cpu 5G -n 2 -c 1 -N 1
#SBATCH --time 10:00:00
#SBATCH --partition=psych_day
#SBATCH --job-name randomise
#SBATCH --output=logs/%j-randomise.out

# Initialise FSL properly
source "$FSLDIR/etc/fslconf/fsl.sh"
module load FSL
export FSLOUTPUTTYPE=NIFTI_GZ

# Move to the right directory
cd "/gpfs/milgram/scratch60/turk-browne/$1/sandbox/GLM_structs/$2/MNI152NLin2009cAsym/" || exit

# Create a merged file of all the 3D maps
OUTPUT_FILE="merged_contrast_$2_all-subs_MNI152NLin2009cAsym.nii.gz"
fslmerge -t "$OUTPUT_FILE" ./contrast*.nii.gz

# Compute the 1 tailed test & save it
FINAL_OUTPUT="randomise_output_$2_MNI152NLin2009cAsym"
randomise -i "$OUTPUT_FILE" -o "$FINAL_OUTPUT" -D -T