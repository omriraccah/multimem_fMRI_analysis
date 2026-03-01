#!/bin/bash
#SBATCH --mem-per-cpu 32G -n 2 -c 1 -N 1
#SBATCH --time 10:00:00
#SBATCH --partition=psych_day
#SBATCH --job-name randomise
#SBATCH --output=logs/%j-randomise.out

# Load FSL module first, then initialize
module load FSL
source $FSLDIR/etc/fslconf/fsl.sh
export FSLOUTPUTTYPE=NIFTI_GZ

# Move to the right directory
if [ "$3" == "condition" ]; then
    cd "/gpfs/milgram/scratch60/turk-browne/$1/sandbox/searchlight_rsa_results/$2/MNI152NLin2009cAsym" || exit
else
    cd "/gpfs/milgram/scratch60/turk-browne/$1/sandbox/searchlight_rsa_results/$3/$2" || exit
fi

# Count only subject files (starting with "sub-")
NUM_SUBJECTS=$(ls -1 sub-*.nii.gz | wc -l)
echo "Found $NUM_SUBJECTS individual subject files (sub-*) going into merged file"

# List the subject files to verify
echo "Subject files found:"
ls -1 sub-*.nii.gz

# Create a merged file of ONLY the subject 3D maps
OUTPUT_FILE="merged_one-tailed_$3_$2_all-subs_MNI152NLin2009cAsym.nii.gz"
fslmerge -t "$OUTPUT_FILE" sub-*.nii.gz

# Create design matrix file (design.mat) in FSL vest format for 30 participants
cat > design.mat << EOF
/NumWaves 1
/NumPoints 30
/Matrix
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
EOF

# Create contrast file (design.con) in FSL vest format
cat > design.con << EOF
/ContrastName1 positive
/NumWaves 1
/NumContrasts 1
/Matrix
1
EOF

# Compute the 1 tailed test & save it
FINAL_OUTPUT="randomise_output_one_tailed_$3_$2_MNI152NLin2009cAsym"
randomise -i "$OUTPUT_FILE" -o "$FINAL_OUTPUT" -d ./design.mat -t ./design.con -T -n 10000