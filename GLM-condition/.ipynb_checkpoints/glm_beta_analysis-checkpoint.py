import nilearn
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
import os
import argparse

# Function to apply mask to betas
def extract_betas(sub, preproc_dir, roi, betas, hpc, pos):
    #if 1:
    #    roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/ASHS/final/func_masks/bin_masks_5B_T1/{pos}_{roi}_mask_T1_non-bin.nii.gz"
    #if 0:
    #    roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/antpost-segs/{roi}_mm{sub}.nii.gz"
        
    if hpc==1:
        roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/ASHS/final/func_masks/bin_masks_5B_T1/{pos}_{roi}_mask_T1_non-bin.nii.gz" 
    # for manual posterior/anterior hippocampal masks
    elif hpc == 0:
        roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/antpost-segs/{roi}_mm{sub}.nii.gz" 
    else:
        roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/harvard-oxford/{roi}.nii.gz"
        

    func_path = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-1_space-T1w_desc-preproc_bold.nii.gz'
    mask_nii = nib.load(roi_mask)
    func_nii = nib.load(func_path)

    # Run resampling here
    resampled_mask_nii = resample_to_img(mask_nii, func_nii, interpolation="nearest")

    # Get array, convert to boolean, mask
    mask_array = resampled_mask_nii.get_fdata()
    mask_bin = mask_array > 0
    masked_betas = betas[mask_bin]

    # if sub == "14" or sub == "19" or sub == "20":
    masked_betas = masked_betas[~np.isnan(masked_betas)]
    masked_betas = np.array(masked_betas)

    return masked_betas


def run_beta_analysis(preproc_dir, beta_dir, conditions, subjects, rois, rois_bool, positions):

    # Dataframe to store results
    df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr'])

    # Loop over conditions
    for condition in conditions:
        # Loop over subjects
        for subject in subjects:

            # Build the path to the computed GLM betas & load them in
            beta_path = f"{beta_dir}/{condition}/T1w/zmap-{condition}_sub-mm{subject}_runs-all_space-T1w.nii.gz"
            beta_img = nib.load(beta_path)
            betas = beta_img.get_fdata()

            # Loop over all the ROIs we want to cover
            for r in range(len(rois)):

                # get the masked betas
                masked_betas = extract_betas(subject, preproc_dir, rois[r], betas, rois_bool[r], positions[r])

                # Average the voxels inside the masked file
                avg_activation = np.mean(masked_betas)

                # Create ROI name
                if rois[r] == "HPC":
                    roi_name = f'{positions[r]}_{rois[r]}'
                else:
                    roi_name = rois[r]

                # Save the resulting value in a df
                df.loc[len(df)] = [subject, condition, roi_name, avg_activation]
                
                print(f"Just completed subject {subject} with ROI {rois[r]} for condition {condition}", flush=True)

    df['roi'] = df['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": 'TP'
    })
    return df



if __name__ == '__main__':
    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str)

    args = parser.parse_args()
    given_sub = args.subject
    
    user = "or62"

    beta_directory = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/GLM_structs"

    preproc_directory = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/preprocessed"

    # ROI specifications
    #rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1", "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1", "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1", "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC"] # Has to be the name the file is saved as

    #rois_bool = [False, False, False, False, False, False, True,
                 #True, True, True, True, True, True, True, True,
                 #True]  # Boolean to indicate if the program should look in ASHS

    #positions = ["", "", "", "", "", "", "combined", "combined", "combined", "combined",
                #"combined", "combined", "combined", "combined", "left",
                #"right"]  # Orientation for ASHS outputs: combined/left/right
    
    rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1", "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1", "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1", "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",  "Superior Temporal Gyrus, posterior division", "Lateral Occipital Cortex, inferior division"]
    
    rois_bool = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    
    positions = ["", "", "", "", "", "", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "left", "right", "", "","",""]

    #subs = ["01", "02", "03", "05", "06", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "18", "19", "20",
    #        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]
    
    subs = [given_sub]

    conditions = ['audio', 'visual', 'congruent', 'incongruent']

    # Run the analysis
    df = run_beta_analysis(preproc_directory, beta_directory, conditions, subs, rois, rois_bool, positions)

    # Save the results as a csv
    final_path = f"{beta_directory}/results/beta_activations_subs-{subs[-1]}.csv"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    df.to_csv(final_path, index=False)
