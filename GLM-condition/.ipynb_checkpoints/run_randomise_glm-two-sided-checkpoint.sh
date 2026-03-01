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

# Smooth the merged file (5mm FWHM = 2.12mm sigma)
# FSL's -s flag takes sigma, not FWHM. sigma = FWHM / 2.355
SMOOTHED_FILE="merged_contrast_$2_all-subs_MNI152NLin2009cAsym_smooth5mm.nii.gz"
fslmaths "$OUTPUT_FILE" -s 2.12 "$SMOOTHED_FILE"

# Count number of subjects
N_SUBJECTS=$(fslval "$SMOOTHED_FILE" dim4)

# Create design matrix (all ones - modeling the group mean)
DESIGN_MAT="design_$2.mat"
printf "/NumWaves\t1\n/NumPoints\t$N_SUBJECTS\n/PPheights\t\t1.000000e+00\n\n/Matrix\n" > "$DESIGN_MAT"
for ((i=1; i<=N_SUBJECTS; i++)); do
    printf "1.000000e+00\n" >> "$DESIGN_MAT"
done

# Create contrast file (two contrasts: +1 and -1 to test both tails)
CONTRAST_FILE="design_$2.con"
printf "/ContrastName1\tpositive\n/ContrastName2\tnegative\n/NumWaves\t1\n/NumContrasts\t2\n/PPheights\t\t1.000000e+00\t1.000000e+00\n/RequiredEffect\t\t1.000\t1.000\n\n/Matrix\n1.000000e+00\n-1.000000e+00\n" > "$CONTRAST_FILE"

# Run two-tailed test using SMOOTHED file
FINAL_OUTPUT="randomise_output_$2_MNI152NLin2009cAsym"
randomise -i "$SMOOTHED_FILE" -o "$FINAL_OUTPUT" -d "$DESIGN_MAT" -t "$CONTRAST_FILE" -T

# Clean up temporary files
rm "$DESIGN_MAT" "$CONTRAST_FILE"