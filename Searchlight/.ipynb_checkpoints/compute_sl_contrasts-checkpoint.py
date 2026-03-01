#!/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem/bin/python

import warnings
import sys
import os
import time
import argparse
import nibabel as nib
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
from brainiak.searchlight.searchlight import Searchlight
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets
from ants import image_read, image_write, apply_transforms
import nilearn
from nilearn.image import resample_to_img
from os.path import join, exists
import ast

# Suppress warnings
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# import functions form RSA script
sys.path.append('..')
#from RSA.run_rsa_w_exclusion import load_glm_single, strip_paths, get_labels, normalise, cross_modal, fisher_z, inverse_fisher_z, run_fisher, get_rsa_matrix


if __name__ == "__main__":
    ############ Parse CL args ###############
    
    sl_dir = "/gpfs/milgram/scratch60/turk-browne/or62/sandbox/preprocessed/searchlight_rsa_results"
    
    subs = []
    
    # define contrasts
    contrasts = ["auditory-visual", "visual-auditory", "congruent-incongruet_A", "congruent-incongruet_V", "congruent-incongruet_mean"]
    
    # loop through contrasts
    for contrast_id, contrast_val in contrasts:
    
        # loop through subjects
        
            # load each subject mni file for each condition
            
            # subtract mni masks
            
            # save contrast mni mask
        
    



if __name__ == "__main__":
    ############ Parse CL args ###############
    # Variables for running the full RSA searchlight analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    parser.add_argument('-cond','--cond', type=str)
    p = parser.parse_args()
    
    # Configuration - adjust these parameters as needed
    preproc_dir = "/gpfs/milgram/scratch60/turk-browne/or62/sandbox/preprocessed"
    subjects = p.sub_id  # initialize subject and condition ids
    conditions = p.cond
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'
    
    # Currently not used
    COMM = MPI.COMM_WORLD
    RANK = COMM.rank
    SIZE = COMM.size
    
    # Searchlight parameters
    sl_rad = 3          # Searchlight radius in voxels
    max_blk_edge = 10    # Maximum block edge
    pool_size = None       # Number of parallel processes

    # Analysis parameters
    apply_exclusion = True
    nruns = 9
    norm = False
    fisher_transform = True
    
    # Output directory
    output_dir = f"{preproc_dir}/searchlight_rsa_results"
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = f"{preproc_dir}/searchlight_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # set output directory (specific)
    if conditions == "V-V":
        output_dir_spec = f'{output_dir}/visual'
    elif conditions == "A-A":
        output_dir_spec = f'{output_dir}/audio'
    elif conditions == "C-C":
        output_dir_spec = f'{output_dir}/congruent'
    elif conditions == "V-IC_V":
        output_dir_spec = f'{output_dir}/incongruent_V'
    elif conditions == "A-IC_A":
        output_dir_spec = f'{output_dir}/incongruent_A'
    else:
        print("output directory unaccounter for -- defaulting to main directory")
        output_dir_spec = output_dir
    
    # Load subject data
    print(f"Loading data for subject {subjects}...")
    betas_4d, condlist, affine, dimensions, whole_brain_mask = load_subject_data(
        preproc_dir, subjects, exclusion_path, apply_exclusion, nruns, norm
    )
    
    # Print condition
    print(f"\nRunning searchlight for condition: {conditions}")
    conds = conditions.split('-')
    
    # Print current subject
    print(f"\nProcessing subject {subjects}...")

    # Setup searchlight for this subject
    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, min_active_voxels_proportion=0.8)

    print(f"Setup searchlight inputs for subject {subjects}:")
    print(f"Input data shape: {betas_4d.shape}")
    print(f"Input mask shape: {whole_brain_mask.shape}")
    print(f"Searchlight radius: {sl_rad}")

    # Distribute the data for this subject only
    sl.distribute([betas_4d], whole_brain_mask)

    # Prepare broadcast variables for this subject
    bcvar = [conds, fisher_transform, condlist]
    sl.broadcast(bcvar)

    # Run searchlight for this subject
    print(f"Running searchlight for subject {subjects}...")
    begin_time = time.time()
    sl_result_sub = sl.run_searchlight(calc_rsa_searchlight, pool_size=pool_size)
    end_time = time.time()

    print(f'Subject {subjects} searchlight duration: {np.round((end_time - begin_time), 2)} seconds')
    print(f'Subject {subjects} result shape: {sl_result_sub.shape}')

    # Save the results !
    
    # First, let's save the results in subject-space

    # Ensure proper data type for NIfTI
    result_map = np.array(sl_result_sub, dtype=np.float32)

    # Replace any remaining NaN values with 0
    result_map = np.nan_to_num(result_map, nan=0.0)

    # Create NIfTI image
    result_nii = nib.Nifti1Image(result_map, affine)
  
    # set output path and save
    output_filename = f"sub-mm{subjects}_cond-{conditions}_searchlight-rsa_space_sub.nii.gz"
    output_dir_spec_t1 = f'{output_dir_spec}/T1w'
    
    output_path_sub_results = join(output_dir_spec_t1, output_filename)
    nib.save(result_nii, output_path_sub_results)
    
    # Now, let's save the results in MNI space
    
    # Path to MNI file
    mni_path = image_read("/home/or62/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
    
    # Build path to the MNI transform
    t1w_to_mni_transform = f'{preproc_dir}/sub-mm{subjects}/anat/sub-mm{subjects}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    
    # Convert the map to MNI space
    result_map_mni = apply_transforms(
        fixed=image_read(mni_path),
        moving=image_read(output_path_sub_results),
        transformlist=t1w_to_mni_transform,
        interpolator='linear'
    )

    # Set output path for MNI image
    output_filename_mni = f"sub-mm{subjects}_cond-{conditions}_searchlight-rsa_space_mni.nii.gz"
    output_dir_spec_t1 = f'{output_dir_spec}/MNI152NLin2009cAsym'
    
    output_path_mni = join(output_dir_spec_t1, output_filename_mni)
    
    # Save the MNI zmap as a NIFTI file
    image_write(result_map_mni, output_path_mni)
    
    print(f"\nCompleted searchlight for all subjects for condition: {conditions}")
    print("Note: Group averaging not performed due to subject-specific spaces.")

    print("\nSearchlight analysis completed!")    


