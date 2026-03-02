#!/bin/bash
#SBATCH --mem-per-cpu 32G -n 2 -c 1 -N 1
#SBATCH --time 10:00:00
#SBATCH --partition=psych_day
#SBATCH --job-name randomise
#SBATCH --output=logs/%j-randomise.out

# Initialise FSL properly
source "$FSLDIR/etc/fslconf/fsl.sh"
module load FSL
export FSLOUTPUTTYPE=NIFTI_GZ

# Move to the right directory
# $3 should be either "contrasts" or "condition"
if [ "$3" == "condition" ]; then
    cd "./MNI152NLin2009cAsym" || exit
else
    cd "./searchlight_rsa_results/$3/$2" || exit
fi

# will need to add MNI152NLin2009cAsym after $2/

# Create a merged file of all the 3D maps
# Count only subject files (starting with "sub-")
NUM_SUBJECTS=$(ls -1 sub-*.nii.gz | wc -l)
echo "Found $NUM_SUBJECTS individual subject files (sub-*) going into merged file"

# List the subject files to verify
echo "Subject files found:"
ls -1 sub-*.nii.gz

# Create a merged file of ONLY the subject 3D maps
OUTPUT_FILE="merged_$3_$2_all-subs_MNI152NLin2009cAsym.nii.gz"
fslmerge -t "$OUTPUT_FILE" sub-*.nii.gz

# Compute the 1 tailed test & save it
FINAL_OUTPUT="randomise_output_$3_$2_MNI152NLin2009cAsym"
# randomise -i "$OUTPUT_FILE" -o "$FINAL_OUTPUT" -1 -T
randomise -i "$OUTPUT_FILE" -o "$FINAL_OUTPUT" -1 -T -n 10000 