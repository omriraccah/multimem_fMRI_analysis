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


# Function to load subject data
def load_subject_data(preproc_dir, sub, exclusion_path, apply_exclusion, nruns=9, norm=False):
    """Load all data for a single subject"""
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
    
    return betas_4d, condlist, affine, dimensions, whole_brain_mask

# Function to make the RSA matrix
"""
def get_rsa_matrix(conds, condlist, masked_betas):
    # Now we need to filter, sort, RSA, and visualise
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]

    # Initialise RSA matrix in pandas
    rsa_corrs = pd.DataFrame(index=scene_id_1, columns=scene_id_2)

    # Create a log on comparisons we have done
    log = []
    
    # OR_edit: added to replace all instances of nruns (in case whole runs were deleted)
    runs = condlist['runs'].unique()
    
    for s in scene_id_1:
        for ss in scene_id_2:
            # Check the log
            if f"{ss}_{s}" in log:
                rsa_corrs.loc[s, ss] = rsa_corrs.loc[ss, s]
                log.append(f"{s}_{ss}")
                continue
            else:
                log.append(f"{s}_{ss}")
            # Get indices for the scenes we want
            if "IC" in conds[0]:
                ind1 = condlist[f"conds_spec_{conds[0]}"].str.contains(f"{s}").to_list()
            else:
                ind1 = condlist['conds_spec'].str.contains(f'{s}').to_list()

            if "IC" in conds[1]:
                ind2 = condlist[f"conds_spec_{conds[1]}"].str.contains(f"{ss}").to_list()
            else:
                ind2 = condlist['conds_spec'].str.contains(f'{ss}').to_list()

            if s != ss:
                # Get the data
                s1_data = masked_betas[ind1]
                s1_labels = condlist[ind1]
                s2_data = masked_betas[ind2]
                s2_labels = condlist[ind2]

                # List to store 81 x 2 correlations
                corrs = []

                # Create folds
                for r in runs: # OR_edit: chnaged from range(nruns) to runs in case any excluded
                    for rr in runs:
                        # Get run indices
                        s1_test_idx = (s1_labels['runs'] == r).to_list()
                        s2_test_idx = (s2_labels['runs'] == rr).to_list()

                        # Extract the test & train data
                        s1_test_data = s1_data[s1_test_idx]
                        s1_train_data = s1_data[[not x for x in s1_test_idx]]
                        s2_test_data = s2_data[s2_test_idx]
                        s2_train_data = s2_data[[not x for x in s2_test_idx]]

                        # Average the train data
                        s1_train_data = np.sum(s1_train_data, axis=0) / (len(condlist['runs'].unique()) - 1)
                        s2_train_data = np.sum(s2_train_data, axis=0) / (len(condlist['runs'].unique()) - 1)
                        
                        corr1 = np.corrcoef(s1_test_data, s2_train_data)
                        corr2 = np.corrcoef(s2_test_data, s1_train_data)

                        # Save the results
                        if corr1 != []: # OR_edit: I added these conditionals. When a trial is excluded the correlation returns an empty matrix. Hence, we simply do not add it to the list of correlations.
                            corrs.append(corr1[0, 1])
                        
                        if corr2 != []:
                            corrs.append(corr2[0, 1])

                # Average all correlations across folds
                mean_corr = np.mean(corrs)
                print("MEAN CORRELATION:")
                print(mean_corr)

                # Calculate correlation and store it
                rsa_corrs.loc[s, ss] = mean_corr

            else:  # Diagonal
                # Extract the data
                s_data = masked_betas[ind1]
                s_labels = condlist[ind1]
                
                #print("SHAPE OF EXTRACTED DATA")
                #print(s_data.shape)
                
                # Store the correlations for each split
                corrs = []
                # Perform reliability
                 # OR_edit: same changes here to account for left out blocks
                for run in runs:
                    # Extract run mask
                    run_idx = (s_labels['runs'] == run).to_list()

                    # Extract test and train data
                    test_data = s_data[run_idx]
                    train_data = s_data[[not x for x in run_idx]]

                    # Average the train data
                    train_data = np.sum(train_data, axis=0) / (len(condlist['runs'].unique()) - 1)

                    # Correlate the train and test data
                    #print("print TRAIN shape")
                    #print(train_data)
                    #print("print TEST shape")
                    #print(test_data)
                    corr = np.corrcoef(train_data, test_data) # OR_edit: same changes here to account for left out trials 
                    #print("print CORR")
                    #print(corr)
                    if corr != []:
                        corrs.append(corr[0, 1])

                # Average the reliability across runs
                mean_corr = np.mean(corrs)

                # Store the reliability measure
                rsa_corrs.loc[s, ss] = mean_corr

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)

    return rsa_corrs
"""

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

    data4D = data[0]
    print("printing data shape:")
    print(data4D.shape)
    
    # Extract data for the current searchlight sphere
    voxel_data = data4D.reshape(-1, data4D.shape[-1])
    voxel_data = voxel_data.T 
    print("PRINTING VOXELS!!")
    print(voxel_data)
    
    # Get RSA matrix for current subject
    rsa_matrix = get_rsa_matrix(conds, condlist, voxel_data)
    rsa_matrix = rsa_matrix.to_numpy()

    if fisher: rsa_matrix = run_fisher(rsa_matrix)

    # Calculate diagonal mean
    mean_diag = np.trace(rsa_matrix) / rsa_matrix.shape[0]
    mean_dian_z = mean_diag
    if fisher: mean_diag = inverse_fisher_z(mean_diag)

    # Calculate off diagonal mean
    mean_off_diag = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
    mean_off_diag_z = mean_off_diag
    
    # Compute off-diag difference as metric (in z-space)
    metric = mean_dian_z - mean_off_diag_z
    
    return metric

    """

def results_sub2mni(result_map, mni_path, preproc_dir):

    t1w_to_mni_transform = f'{preproc_dir}/sub-mm{subjects}/anat/sub-mm{subjects}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'

    
    result_map_mni = apply_transforms(
        fixed=mni_path,
        moving=result_map,
        transformlist=t1w_to_mni_transform,
        interpolator='nearestNeighbor'
    )
    
    return result_map_mni

    """
    """

# Main function for running RSA searchlight
def run_rsa_searchlight_analysis(preproc_dir, subjects, conditions, exclusion_path, 
                                apply_exclusion=True, nruns=9, norm=False, 
                                fisher_transform=True, sl_rad=3, max_blk_edge=10, 
                                pool_size=1, mni_path=None, output_dir=None): 
    
    Run RSA searchlight analysis across multiple subjects and conditions
    
    Parameters:
    - preproc_dir: Path to preprocessed data directory
    - subjects: List of subject IDs
    - conditions: List of condition pairs (e.g., ['V-A', 'A-A', 'V-V'])
    - exclusion_path: Path to exclusion criteria file
    - apply_exclusion: Whether to apply exclusion criteria
    - nruns: Number of runs
    - norm: Whether to normalize betas
    - fisher_transform: Whether to apply Fisher z-transform
    - sl_rad: Searchlight radius in voxels
    - max_blk_edge: Maximum block edge for searchlight
    - pool_size: Number of processes for parallel processing
    - output_dir: Directory to save results
    
    
    if output_dir is None:
        output_dir = f"{preproc_dir}/searchlight_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for all subjects
    #print("Loading subject data...")
  
    if RANK==0:
        
        print(f"Loading data for subject {subjects}...")
        betas_4d, condlist, affine, dimensions, whole_brain_mask = load_subject_data(
            preproc_dir, subjects, exclusion_path, apply_exclusion, nruns, norm
        )
        
    else: 
        betas_4d = None 
    
    # print condition
    print(f"\nRunning searchlight for condition: {conditions}")
    conds = conditions.split('-')

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

    # Since we're running one subject at a time, sl_result_sub is 1D
    result_map = sl_result_sub

    # Ensure proper data type for NIfTI
    result_map = np.array(result_map, dtype=np.float32)

    # Replace any remaining NaN values with 0
    result_map = np.nan_to_num(result_map, nan=0.0)

    # Create NIfTI image
    result_nii = nib.Nifti1Image(result_map, affine)

    # create MNI space version of result_map
    result_mni = results_sub2mni(result_map, mni_path, preproc_dir)
    # Create NIfTI image
    result_nii_mni = nib.Nifti1Image(result_mni, None)

    # Save the result
    output_filename = f"sub-mm{subjects}_cond-{cond_pair}_searchlight-rsa_sub_space.nii.gz"
    output_filename_mni = f"sub-mm{subjects}_cond-{cond_pair}_searchlight-rsa_mni_scace.nii.gz"
    output_path = join(output_dir, output_filename)
    output_path_mni = join(output_dir, output_filename)
    nib.save(result_nii, output_path)
    nib.save(result_nii_mni, output_path_mni)
    print(f"Saved: {output_path}")

    # Store result for potential group analysis
    # all_results.append(result_map)

    print(f"\nCompleted searchlight for all subjects for condition: {cond_pair}")
    print("Note: Group averaging not performed due to subject-specific spaces.")
    
    print("\nSearchlight analysis completed!")

    """

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

    mni_path = image_read("/home/or62/nilearn_data/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
    
    # Output directory
    output_dir = f"{preproc_dir}/searchlight_rsa_results"

    
    if output_dir is None:
        output_dir = f"{preproc_dir}/searchlight_results"
    os.makedirs(output_dir, exist_ok=True)
    
    #print(f"Loading data for subject {subjects}...")
    betas_4d, condlist, affine, dimensions, whole_brain_mask = load_subject_data(
        preproc_dir, subjects, exclusion_path, apply_exclusion, nruns, norm
    )
  
    if RANK==0:
        
        print(f"Loading data for subject {subjects}...")
        betas_4d, condlist, affine, dimensions, whole_brain_mask = load_subject_data(
            preproc_dir, subjects, exclusion_path, apply_exclusion, nruns, norm
        )
        print("PRINT RANK:")
        print(RANK)
        
    else: 
        betas_4d[:,:,:,:] = None
        print("PRINT RANK:")
        print(RANK)
    
    # print condition
    print(f"\nRunning searchlight for condition: {conditions}")
    conds = conditions.split('-')

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

    # Save the result
    if RANK==0:
        # Since we're running one subject at a time, sl_result_sub is 1D
        result_map = sl_result_sub

        # Ensure proper data type for NIfTI
        result_map = np.array(result_map, dtype=np.float32)

        # Replace any remaining NaN values with 0
        result_map = np.nan_to_num(result_map, nan=0.0)

        # Create NIfTI image
        result_nii = nib.Nifti1Image(result_map, affine)

        # create MNI space version of result_map
        result_mni = results_sub2mni(result_map, mni_path, preproc_dir)
        # Create NIfTI image
        #result_nii_mni = nib.Nifti1Image(result_mni, affine)

        output_filename = f"sub-mm{subjects}_cond-{conditions}_searchlight-rsa_sub_space_WITHOUT_RANKS.nii.gz"
        output_filename_mni = f"sub-mm{subjects}_cond-{conditions}_searchlight-rsa_mni_scace.nii.gz"
        output_path = join(output_dir, output_filename)
        output_path_mni = join(output_dir, output_filename)
        nib.save(result_nii, output_path)
        #image_write(result_mni, output_path_mni)
        #print(f"Saved: {output_path}")

        # Store result for potential group analysis
        # all_results.append(result_map)

        print(f"\nCompleted searchlight for all subjects for condition: {conditions}")
        print("Note: Group averaging not performed due to subject-specific spaces.")

        print("\nSearchlight analysis completed!")    
    

    