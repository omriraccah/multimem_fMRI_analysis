import numpy as np
import nilearn as ni
import nibabel as nib
import pandas as pd
import os
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from ants import image_read, image_write, apply_transforms
import argparse
import ast
from nilearn.plotting import plot_design_matrix

# Helper to strip the paths of the stimuli to create numbered conditions for GLMSingle
def strip_paths(list_o_paths):
    stripped = []
    for s in list_o_paths:
        if 'Incongruent' not in s:
            stripped.append([int(s.split('/S')[-1].split('_')[0])])
        else:
            temp = [int(s.split('/S')[-1].split('_')[i].replace('S', '')) for i in [0, 1]]
            stripped.append(temp)
    return stripped

# Function to make the design matrices
def make_design_matrices(func_dir, sub, nruns, conds):
    """
    :param func_dir: String path to preprocessed data
    :param sub: String subject number "mm01"
    :param nruns: Int number of runs
    :param conds: List of conditions: "Audio", "Visual", "Congruent", "Incongruent"
    :return: List of nruns design matrices
    """

    # initialize list for storing matrices across runs
    designmats = []
    
    # exlusion df
    exclusion_df = pd.read_csv('/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv')
    exclusion_df_sub = exclusion_df[exclusion_df['subs'] == f"mm{sub}"]

    # Loop over runs and process events files, turn into design matrices, append
    for run in range(1, nruns + 1):
        # load events data for current run
        events_path = f"{func_dir}/sub-mm{sub}_task-multisensorymemory_run-{run}_events.tsv"
        events = pd.read_csv(events_path, sep='\t')

        # load confounds file for current run
        confounds, _ = ni.interfaces.fmriprep.load_confounds(f"{func_dir}/sub-mm{sub}_task-multisensorymemory_run-{run}_space-T1w_desc-preproc_bold.nii.gz", strategy=['high_pass', 'motion', 'scrub'], motion='basic', scrub=0, fd_threshold=0.5, std_dvars_threshold=1.5)
        
        # load exlusion trials for current run
        exclusion_df_sub_run = exclusion_df_sub[exclusion_df_sub['run'] == run]
        
        if exclusion_df_sub_run['stim_past_thr'].empty == False:
            trials_to_exclude = ast.literal_eval(exclusion_df_sub_run['stim_past_thr'].iloc[0])
        else:
            trials_to_exclude = []

        # adapt trial type labels to reflect conditions
        for index, row in events.iterrows():
            if '/Audio/' in events['trial_type'][index]:
                
                num = strip_paths([events['trial_type'][index][2:-2]])
                num = num[0][0]
                trial_name = f"S{num}_A"
                print
                if trial_name in trials_to_exclude:
                    events.loc[index, 'trial_type'] = "Excluded"
                else:
                    events.loc[index, 'trial_type'] = "Audio"
                    
            elif '/Visual/' in events['trial_type'][index]:
                num = strip_paths([events['trial_type'][index][2:-2]])
                num = num[0][0]
                trial_name = f"S{num}_V"
                
                if trial_name in trials_to_exclude:
                    events.loc[index, 'trial_type'] = "Excluded"
                else:
                    events.loc[index, 'trial_type'] = "Visual"
                    
            elif '/Congruent/' in events['trial_type'][index]:
                num = strip_paths([events['trial_type'][index][2:-2]])
                num = num[0][0]
                trial_name = f"S{num}_C"
                
                if trial_name in trials_to_exclude:
                    events.loc[index, 'trial_type'] = "Excluded"
                else:
                    events.loc[index, 'trial_type'] = "Congruent"
                    
            elif '/Incongruent/' in events['trial_type'][index]:
                num = strip_paths([events['trial_type'][index][2:-2]])
                num = num[0]
                trial_name = f"S{num[0]}_S{num[1]}_IC"
                
                if trial_name in trials_to_exclude:
                    events.loc[index, 'trial_type'] = "Excluded"
                else:
                    events.loc[index, 'trial_type'] = "Incongruent"

        # define conditions for GLM analysis
        events = events[events['trial_type'].isin(conds)]

        # change all durations to six seconds to account for slight discrepancies in task readout
        events['duration'] = events['duration'].replace(events['duration'].unique(), 6)

        # Set Frame Times
        t_r = 1.5
        n_scans = 337 if sub == "01" else 335
        frame_times = (np.arange(n_scans) * t_r)

        # create first-level design matrix
        designmat = make_first_level_design_matrix(
            frame_times,
            events,
            drift_model=None,  # loading it from fmriprep instead
            # drift_order=3,
            add_regs=confounds,
            # high_pass=high_pass,
            hrf_model="glover + derivative + dispersion")

        designmats.append(designmat)

    return designmats


# Function to load the BOLD data for all the runs for one subject
def load_sub_bold(func_dir, sub, nruns):
    fmri_imgs = []
    for run in range(1, nruns + 1):
        filename = f'{func_dir}/sub-mm{sub}_task-multisensorymemory_run-{run}_space-T1w_desc-preproc_bold.nii.gz'
        fmri_imgs.append(ni.image.load_img(filename))

    return fmri_imgs


# Function to estimate glm contrasts for one subject all runs
def compute_glm_contrasts(preproc_dir, sub, nruns, conds, mni_path, save_dir):

    # Get the design matrices
    design_matrices = make_design_matrices(f"{preproc_dir}/sub-mm{sub}/func/", sub, nruns, conds)

    # Get the functional data
    bold_data = load_sub_bold(f"{preproc_dir}/sub-mm{sub}/func/", sub, nruns)
    
    brain_mask_path = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-1_space-T1w_desc-brain_mask.nii.gz'

    # Initialise the first level model
    fmri_glm = FirstLevelModel(
        slice_time_ref=0.5,  # IMPORTANT, when using slicetiming in fmriprep
        noise_model="ar1",
        smoothing_fwhm=None,
        mask_img=brain_mask_path)  # Applied spatial smoothing for clearer images in this initial analysis

    # Estimate the beta coefficients
    fmri_glm = fmri_glm.fit(bold_data, design_matrices=design_matrices)

    # create identity matrix over which to define contrasts
    contrast_matrix = np.eye(design_matrices[0].shape[1])
    basic_contrasts = {column: contrast_matrix[i] for i, column in enumerate(design_matrices[0].columns)}

    # specify contrasts
    contrasts = {
        "visual": basic_contrasts["Visual"],
        "audio": basic_contrasts["Audio"],
        "congruent": basic_contrasts["Congruent"],
        "incongruent": basic_contrasts["Incongruent"],
        "visual-audio": basic_contrasts["Visual"] - basic_contrasts["Audio"],
        "audio-visual": basic_contrasts["Audio"] - basic_contrasts["Visual"],
        "congruent-incongruent": basic_contrasts["Congruent"] - basic_contrasts["Incongruent"],
        "incongruent-congruent": basic_contrasts["Incongruent"] - basic_contrasts["Congruent"],
        # New contrasts added below
        "congruent-unisensory": basic_contrasts["Congruent"] - np.mean([basic_contrasts["Audio"], basic_contrasts["Visual"]], axis=0),
        "congruent-audio": basic_contrasts["Congruent"] - basic_contrasts["Audio"],
        "congruent-visual": basic_contrasts["Congruent"] - basic_contrasts["Visual"]
    }

    # Calculate the FIXED EFFECTS MAP and then z-score the results. The map is across runs, 3D with voxel resolution.
    for contrast_id, contrast_val in contrasts.items():
        # Calculate the map (returns a NIFTI object)
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="effect_size")

        # Check if this is a contrast or a normal map
        if '-' in contrast_id:
            # Path to save the T1w file at, make it if it doesn't exist
            t1w_final_path = f"{save_dir}/{contrast_id}/T1w/contrast-{contrast_id}_sub-mm{sub}_runs-all_space-T1w.nii.gz"
            os.makedirs(os.path.dirname(t1w_final_path), exist_ok=True)

            # Save the T1w zmap as a NIFTI file
            z_map.to_filename(t1w_final_path)

            # Build path to the MNI transform
            t1w_to_mni_transform = f'{preproc_dir}/sub-mm{sub}/anat/sub-mm{sub}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'

            # Convert the map to MNI space
            result_map_mni = apply_transforms(
                fixed=image_read(mni_path),
                moving=image_read(t1w_final_path),
                transformlist=t1w_to_mni_transform,
                interpolator='linear'
            )

            # Path to save the MNI file at, make it if it doesn't exist
            mni_final_path = f"{save_dir}/{contrast_id}/MNI152NLin2009cAsym/contrast-{contrast_id}_sub-mm{sub}_runs-all_space-MNI152NLin2009cAsym.nii.gz"
            os.makedirs(os.path.dirname(mni_final_path), exist_ok=True)

            # Save the MNI zmap as a NIFTI file
            image_write(result_map_mni, mni_final_path)
        else:
            # Not a contrast just a normal map - save and don't convert to MNI

            # Path to save the T1w file at, make it if it doesn't exist
            t1w_final_path = f"{save_dir}/{contrast_id}/T1w/zmap-{contrast_id}_sub-mm{sub}_runs-all_space-T1w.nii.gz"
            os.makedirs(os.path.dirname(t1w_final_path), exist_ok=True)

            # Save the T1w zmap as a NIFTI file
            z_map.to_filename(t1w_final_path)
        
        print(f"Successfully completed contrast/map: {contrast_id} for subject: mm{sub}", flush=True)

    print(f"Successfully completed all contrasts for subject: mm{sub}", flush=True)

    return 1


# Function to run the GLM analysis for all subjects
def run_glm_analysis(preproc_dir, subs, nruns, conds, mni_path, save_dir):

    # Loop over subjects
    for sub in subs:

        # Run the GLM & save the contrasts in the relevant directory
        compute_glm_contrasts(preproc_dir, sub, nruns, conds, mni_path, save_dir)

    return 1


if __name__ == "__main__":
    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str)

    args = parser.parse_args()
    given_sub = args.subject

    user = "or62"

    #subs = ["01", "02", "03", "05", "06", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "18", "19", "20",
    #        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]
    subs = [given_sub]
    
    nruns = 9

    conditions = ["Audio", "Visual", "Congruent", "Incongruent"]

    mni_path = "/home/or62/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"

    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/preprocessed"

    final_save_path = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/GLM_structs/"

    run_glm_analysis(preproc_dir, subs, nruns, conditions, mni_path, final_save_path)