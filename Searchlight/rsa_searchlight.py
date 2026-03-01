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
from RSA.run_rsa_w_exclusion import load_glm_single, strip_paths, get_labels, normalise, cross_modal, fisher_z, inverse_fisher_z, run_fisher, get_rsa_matrix

# Function to load subject data - modified for MPI
def load_subject_data(preproc_dir, sub, exclusion_path, apply_exclusion, nruns=9, norm=False, rank=0):
    """Load all data for a single subject - only on rank 0"""
    
    if rank == 0:
        print(f"Loading data for subject {sub} on rank {rank}...")
        
        # Load GLM single betas
        if apply_exclusion:
            exclusion_df = pd.read_csv(exclusion_path)
            if f'mm{sub}' in exclusion_df['subs'].values:
                betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
            else: 
                betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
        else:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
        
        # Get condition labels
        condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
        
        # Load a reference functional image to get dimensions and affine
        func_path = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-1_space-T1w_desc-preproc_bold.nii.gz'
        func_nii = nib.load(func_path)
        affine = func_nii.affine
        dimensions = func_nii.shape[:3]
        
        # Reshape betas to 4D (x, y, z, trials)
        betas_4d = betas#.reshape(dimensions + (betas.shape[0],))
        
        # Normalize if requested
        if norm:
            betas_reshaped = betas.T  # Shape: (trials, voxels)
            betas_normalized = normalise(betas_reshaped, condlist)
            betas_4d = betas_normalized.T.reshape(dimensions + (betas.shape[0],))
        
        # load whole-brain mask
        wb_mask_path = f'{preproc_dir}/sub-mm{sub}/anat/sub-mm{sub}_desc-brain_mask.nii.gz'
        mask_nii = nib.load(wb_mask_path)
        
        # Run resampling here
        resampled_mask_nii = resample_to_img(mask_nii, func_nii, interpolation="nearest")

        # Get array
        mask_array = resampled_mask_nii.get_fdata()

        # convert to boolean
        whole_brain_mask = mask_array > 0
        
        print(f'number of total voxels in mask (entire mask): {np.sum(whole_brain_mask)}')
        
        # small mask (UNCOMMENT TO TEST CODE)
        #whole_brain_mask[:,:,:] = 0
        #whole_brain_mask[0:10,0:10,0:10] = 1
        
        print(f'number of total voxels in mask (after small mask): {np.sum(whole_brain_mask)}')
        
        return betas_4d, condlist, affine, dimensions, whole_brain_mask
    
    else:
        # Other ranks don't load the full data, but need basic info
        return None, None, None, None, None

def load_metadata_all_ranks(preproc_dir, sub, exclusion_path, apply_exclusion, nruns=9):
    """Load metadata that all ranks need (labels and mask info)"""
    
    # Get condition labels (needed by all ranks)
    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
    
    # Load reference functional image for dimensions and affine (needed by all ranks)
    func_path = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-1_space-T1w_desc-preproc_bold.nii.gz'
    func_nii = nib.load(func_path)
    affine = func_nii.affine
    dimensions = func_nii.shape[:3]
    
    # Load and process mask (needed by all ranks)
    wb_mask_path = f'{preproc_dir}/sub-mm{sub}/anat/sub-mm{sub}_desc-brain_mask.nii.gz'
    mask_nii = nib.load(wb_mask_path)
    resampled_mask_nii = resample_to_img(mask_nii, func_nii, interpolation="nearest")
    mask_array = resampled_mask_nii.get_fdata()
    whole_brain_mask = mask_array > 0
    
    print(f'numnber of total voxels in mask (entire mask): {np.sum(whole_brain_mask)}')
    
    # Apply same small mask for testing (UNCOMMENT TO TEST CODE)
    #whole_brain_mask[:,:,:] = 0
    #whole_brain_mask[0:10,0:10,0:10] = 1
    
    print(f'numnber of total voxels in mask (after small mask): {np.sum(whole_brain_mask)}')
    
    return condlist, affine, dimensions, whole_brain_mask

# Kernel for RSA analysis
def calc_rsa_searchlight(data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel function for RSA analysis 
    
    Parameters:
    - data: list of 4D arrays (one per subject) containing beta values
    - sl_mask: boolean mask for current searchlight sphere
    - myrad: searchlight radius (not used in this implementation)
    - bcvar: broadcasted variables containing condition information
    
    Returns:
    - Array of RSA similarity values for each subject
    """
    results = []
    
    # Extract conditions and analysis type from broadcast variables
    # bcvar contains: [conds, fisher_transform, condlist]
    conds = bcvar[0]
    fisher = bcvar[1]
    condlist = bcvar[2]
    
    # # TO ASK ERICA. WHY DO I NEED THE BELOW:
    # if np.sum(sl_mask) < myrad*3*2: 
    #     print(f'TOO SMALL')
    #     return np.nan

    data4D = data[0]
    print("printing data shape: ")
    print(data4D.shape)
    
    # Extract data for the current searchlight sphere
    voxel_data = data4D.reshape(-1, data4D.shape[-1])
    voxel_data = voxel_data.T 
    print(f'after reshaping for this SL, shape is {voxel_data.shape} | should be n_trials x n_voxels_in_sl')
    
    # Get RSA matrix for current subject
    rsa_matrix = get_rsa_matrix(conds, condlist, voxel_data)
    rsa_matrix = rsa_matrix.to_numpy()
    print(f'rsa_matrix is shape {rsa_matrix.shape}')
    #if fisher: rsa_matrix = run_fisher(rsa_matrix)

    # Calculate diagonal mean
    mean_diag = np.trace(rsa_matrix) / rsa_matrix.shape[0]
    mean_dian_z = mean_diag
    # if fisher: mean_diag = inverse_fisher_z(mean_diag)

    # Calculate off diagonal mean
    mean_off_diag = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
    mean_off_diag_z = mean_off_diag
    
    # Compute off-diag difference as metric (in z-space)
    metric = mean_dian_z - mean_off_diag_z
    print(f'returning metric: {metric}')
    
    return metric
    

if __name__ == "__main__":
    # Initialize MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.rank
    SIZE = COMM.size
    
    print(f"Rank {RANK} of {SIZE} processes started")
    
    ############ Parse CL args ###############
    # Variables for running the full RSA searchlight analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    parser.add_argument('-cond','--cond', type=str)
    p = parser.parse_args()
    
    # Configuration - adjust these parameters as needed
    preproc_dir = "/gpfs/milgram/scratch60/turk-browne/or62/sandbox/preprocessed"
    sandbox_dir = "/gpfs/milgram/scratch60/turk-browne/or62/sandbox"
    subjects = p.sub_id  # initialize subject and condition ids
    conditions = p.cond
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'
    
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
    output_dir = f"{sandbox_dir}/searchlight_rsa_results"
    
    # Create output directory if it doesn't exist (only on rank 0)
    if RANK == 0:
        if output_dir is None:
            output_dir = f"{sandbox_dir}/searchlight_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # set output directory (specific)
        if conditions == "V-V":
            cond_label = "visual"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "A-A":
            cond_label = "auditory"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "C-C":
            cond_label = "congruent"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "V-IC_V":
            cond_label = "incongruent_V"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "A-IC_A":
            cond_label = "incongruent_A"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "V-A":
            cond_label = "cross_visual_auditory"
            output_dir_spec = f'{output_dir}/{cond_label}'
        else:
            cond_label = "NAN"
            print("output directory unaccounted for -- defaulting to main directory")
            output_dir_spec = output_dir
        os.makedirs(output_dir_spec, exist_ok=True)
        print("All output paths created!")
    else:
        # Non-zero ranks need these variables too
        if conditions == "V-V":
            cond_label = "visual"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "A-A":
            cond_label = "auditory"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "C-C":
            cond_label = "congruent"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "V-IC_V":
            cond_label = "incongruent_V"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "A-IC_A":
            cond_label = "incongruent_A"
            output_dir_spec = f'{output_dir}/{cond_label}'
        elif conditions == "V-A":
            cond_label = "cross_visual_auditory"
            output_dir_spec = f'{output_dir}/{cond_label}'
        else:
            cond_label = "NAN"
            output_dir_spec = output_dir
    
    # Wait for rank 0 to create directories
    COMM.barrier()
    
    # Load data efficiently based on rank
    if RANK == 0:
        # Only rank 0 loads the full data
        print(f"Loading data for subject {subjects} on rank {RANK}...")
        betas_4d, condlist, affine, dimensions, whole_brain_mask = load_subject_data(
            preproc_dir, subjects, exclusion_path, apply_exclusion, nruns, norm, RANK
        )
        data = betas_4d
        print(f'betas shape {np.shape(betas_4d)} | n_conditions: {len(condlist)} | wb mask sum: {np.sum(whole_brain_mask)}')
    else:
        # Other ranks load only metadata
        condlist, affine, dimensions, whole_brain_mask = load_metadata_all_ranks(
            preproc_dir, subjects, exclusion_path, apply_exclusion, nruns
        )
        data = None
        print(f"Rank {RANK}: Loaded metadata only")
    
    # Print condition (all ranks)
    if RANK == 0:
        print(f"\nRunning searchlight for condition: {conditions}")
    conds = conditions.split('-')
    
    # Print current subject (all ranks)
    if RANK == 0:
        print(f"\nProcessing subject {subjects}...")

    # Setup searchlight for this subject
    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge, min_active_voxels_proportion=0.8)

    if RANK == 0:
        print(f"Setup searchlight inputs for subject {subjects}:")
        print(f"Input data shape: {betas_4d.shape}")
        print(f"Input mask shape: {whole_brain_mask.shape}")
        print(f"Searchlight radius: {sl_rad}")

    # Distribute the data for this subject only
    # This handles the MPI distribution automatically
    sl.distribute([data], whole_brain_mask)

    # Prepare broadcast variables for this subject
    bcvar = [conds, fisher_transform, condlist]
    sl.broadcast(bcvar)

    # Run searchlight for this subject
    if RANK == 0:
        print(f"Running searchlight for subject {subjects}...")
        begin_time = time.time()
    
    sl_result_sub = sl.run_searchlight(calc_rsa_searchlight, pool_size=pool_size)
    
    if RANK == 0:
        end_time = time.time()
        print(f'Subject {subjects} searchlight duration: {np.round((end_time - begin_time), 2)} seconds')
        print(f'Subject {subjects} result shape: {sl_result_sub.shape}')

        # Save the results only on rank 0
        
        # First, let's save the results in subject-space

        # Ensure proper data type for NIfTI
        result_map = np.array(sl_result_sub, dtype=np.float32)

        # Replace any remaining NaN values with 0
        result_map = np.nan_to_num(result_map, nan=0.0)

        # Create NIfTI image
        result_nii = nib.Nifti1Image(result_map, affine)
      
        # set output path and save
        output_filename = f"sub-mm{subjects}_cond-{cond_label}_searchlight-rsa_space_sub.nii.gz"
        output_dir_spec_t1 = f'{output_dir_spec}/T1w'
        os.makedirs(output_dir_spec_t1, exist_ok=True)
        
        output_path_sub_results = join(output_dir_spec_t1, output_filename)
        nib.save(result_nii, output_path_sub_results)
        print("Results saved in T1w space")
        
        # Now, let's save the results in MNI space
        
        # Path to MNI file
        mni_path = "/home/or62/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
        
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
        output_filename_mni = f"sub-mm{subjects}_cond-{cond_label}_searchlight-rsa_space_mni.nii.gz"
        output_dir_spec_mni = f'{output_dir_spec}/MNI152NLin2009cAsym'
        os.makedirs(output_dir_spec_mni, exist_ok=True)
        
        output_path_mni = join(output_dir_spec_mni, output_filename_mni)
        
        # Save the MNI zmap as a NIFTI file
        image_write(result_map_mni, output_path_mni)
        print("Results saved in MNI space")
        
        print(f"\nCompleted searchlight for all subjects for condition: {conditions}")
        print("Note: Group averaging not performed due to subject-specific spaces.")

        print("\nSearchlight analysis completed!")
    
    # Ensure all ranks finish before exiting
    COMM.barrier()
    
    if RANK == 0:
        print("All ranks completed successfully!")