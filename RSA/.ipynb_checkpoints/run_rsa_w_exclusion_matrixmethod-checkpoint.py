#!/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem/bin/python
import warnings
import sys

from webencodings import labels

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import nibabel as nib
import numpy as np
import nilearn as ni
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from nilearn.image import resample_to_img
from os.path import join, exists, split
import itertools
import argparse
import time
import ast

sns.set(style='white', context='notebook', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")


# Load GLM Single results for the final model
def load_glm_single(outputdir_glmsingle):
    results_glmsingle = dict()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                         allow_pickle=True).item()

    betas = results_glmsingle['typed']['betasmd']

    return betas

# function to strip the paths of the stimuli to create numbered conditions for GLMSingle
def strip_paths(list_o_paths):
    stripped = []
    for s in list_o_paths:
        if 'Incongruent' not in s:
            stripped.append([int(s.split('/S')[-1].split('_')[0])])
        else:
            temp = [int(s.split('/S')[-1].split('_')[i].replace('S', '')) for i in [0, 1]]
            stripped.append(temp)
    return stripped


# Function to create (non-convolved) design matrices based on events files
def get_labels(preproc_dir, sub, exclusion_path, apply_exclusion, nruns="9"):

    # initialize list for storing matrices across runs
    trial_labels = pd.DataFrame()
    trial_labels_cond = pd.DataFrame()
    trial_labels_IC_V = pd.DataFrame()
    trial_labels_IC_A = pd.DataFrame()

    # initialize list for storing run labels
    run_labels = list()
    
    # load exclusion
    exclusion_df = pd.read_csv(exclusion_path)
    
    # get exclusion trials for current subject
    exclusion_df_sub = exclusion_df[exclusion_df["subs"] == f'mm{sub}']

    # counter for runs (given that some may excluded)
    run_counter = 1
        
    for run in range(1,nruns+1):

        # load events data for current run
        events_path = preproc_dir + f'sub-mm{sub}_task-multisensorymemory_run-{run}_events.tsv'
        events = pd.read_csv(events_path, sep='\t')

        # change all durations to six seconds to account for slight discrepecies in task readout
        events['duration'] = events['duration'].replace(events['duration'].unique(), 1.5)

        # remove any ITI events from matrix (slightly different for mm01 due to label issue)
        if sub == "01":
            events = events[~events['trial_type'].str.contains('MCOS')]
            events = events.reset_index(drop=True)
        else:
            events = events[~events['trial_type'].str.contains('ITI')]
            events = events.reset_index(drop=True)

        events["trial_type_label"] = events["trial_type"]
        
        # initialize trials to exclude 
        trials_to_exclude = []
        
        # get exclusion data for current run
        if apply_exclusion:
            exclusion_df_sub_run = exclusion_df_sub[exclusion_df_sub['run'] == run]
            if exclusion_df_sub_run['stim_past_thr'].empty == False:
                trials_to_exclude = ast.literal_eval(exclusion_df_sub_run['stim_past_thr'].iloc[0])
            
        # set variable for exclusion
        exclude_num = 1

        for index, row in events.iterrows():

            if '/Audio/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_A"
                events.loc[index, 'trial_type_cond'] = "A"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"
                
                # if trial should be excluded
                if f"S{num}_A" in trials_to_exclude:
                    events.loc[index, 'trial_type'] = f"exclude_{exclude_num}"
                    exclude_num = 2

            elif '/Visual/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_V"
                events.loc[index, 'trial_type_cond'] = "V"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"
                
                # if trial should be excluded
                if f"S{num}_V" in trials_to_exclude:
                    events.loc[index, 'trial_type'] = f"exclude_{exclude_num}"
                    exclude_num = 2

            elif '/Congruent/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_C"
                events.loc[index, 'trial_type_cond'] = "C"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"
                
                # if trial should be excluded
                if f"S{num}_C" in trials_to_exclude:
                    events.loc[index, 'trial_type'] = f"exclude_{exclude_num}"
                    exclude_num = 2

            elif '/Incongruent/' in events['trial_type_label'][index]:
                
                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0]
                events.loc[index, 'trial_type'] = f"S{num[0]}_S{num[1]}_IC"
                events.loc[index, 'trial_type_IC_V'] = f"S{num[0]}_IC_V"
                events.loc[index, 'trial_type_IC_A'] = f"S{num[1]}_IC_A"
                events.loc[index, 'trial_type_cond'] = "IC"
                
                 # if trial should be excluded
                if f"S{num[0]}_S{num[1]}_IC" in trials_to_exclude:
                    events.loc[index, 'trial_type'] = f"exclude_{exclude_num}"
                    events.loc[index, 'trial_type_IC_V'] = f"excluded_IC"
                    events.loc[index, 'trial_type_IC_A'] = f"excluded_IC"
                    events.loc[index, 'trial_type_cond'] = "IC"
                    exclude_num = 2
            
            # only append block if less than three trials have sig. movement
            if len(trials_to_exclude) < 3:
                run_labels.append(run)

        # save run labels
        if len(trials_to_exclude) < 3:
            trial_labels[run_counter] = events['trial_type']
            trial_labels_IC_V[run_counter] = events['trial_type_IC_V']
            trial_labels_IC_A[run_counter] = events['trial_type_IC_A']
            trial_labels_cond[run_counter] = events['trial_type_cond']
            run_counter = run_counter + 1

    # stack all trials to one column (vertically) for each correspondence to GLM trial betas
    trial_labels = pd.concat([trial_labels, trial_labels.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    trial_labels_cond = pd.concat([trial_labels_cond, trial_labels_cond.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    
    trial_labels_IC_V = pd.concat([trial_labels_IC_V, trial_labels_IC_V.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    trial_labels_IC_A = pd.concat([trial_labels_IC_A, trial_labels_IC_A.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    
    condlist = pd.DataFrame()
    condlist['conds_spec'] = trial_labels['all_runs']
    condlist['conds_gen'] = trial_labels_cond['all_runs']
    condlist['runs'] = run_labels
    condlist['conds_spec_IC_V'] = trial_labels_IC_V['all_runs']
    condlist['conds_spec_IC_A'] = trial_labels_IC_A['all_runs']

    return condlist


# Function to apply mask to betas
def extract_betas(sub, preproc_dir, roi, betas, labels, hpc, pos, norm, atlas):
    if hpc==1:
        if atlas == "ASHS":
            roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/{atlas}/final/func_masks/bin_masks_5B_T1/{pos}_{roi}_mask_T1_non-bin.nii.gz" 
        elif atlas == "ASHS_51":
            roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/{atlas}/final/func_masks/bin_masks_5B_T1/{roi}_non-bin.nii.gz" 
    
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
    masked_betas = betas[mask_bin].T
    
    masked_betas = [row[~np.isnan(row)] for row in masked_betas]
    masked_betas = np.array(masked_betas)

    if norm:
        masked_betas = normalise(masked_betas, labels)

    return masked_betas


# Function to normalise the betas
def normalise(masked_betas, labels):

    normalized_betas = np.zeros_like(masked_betas)
    grouped = labels.groupby(["runs", "conds_gen"])

    for (run, cond), indices in grouped.groups.items():
        # Extract the indices for this group
        group_indices = list(indices)

        # Select the beta values corresponding to this group
        group_betas = masked_betas[group_indices]

        # Normalize within this group (z-score normalization)
        scaler = StandardScaler()
        normalized_betas[group_indices] = scaler.fit_transform(group_betas)

    return normalized_betas

# Function for Fisher transformations
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def run_fisher(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
                matrix[i][j] = fisher_z(matrix[i][j])
    return matrix


# =============================================================================
# VECTORIZED RSA FUNCTIONS
# =============================================================================

def zscore_rows(X):
    """Z-score normalize each row of a matrix (each pattern across voxels)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    return (X - mean) / std


def correlation_matrix(X, Y):
    """
    Compute correlation matrix between rows of X and rows of Y using matrix multiplication.
    X: (n, voxels) - n patterns
    Y: (m, voxels) - m patterns
    Returns: (n, m) correlation matrix
    """
    # Z-score normalize
    X_z = zscore_rows(X)
    Y_z = zscore_rows(Y)
    
    # Correlation via dot product of z-scored vectors
    n_voxels = X.shape[1]
    corr_matrix = np.dot(X_z, Y_z.T) / n_voxels
    
    return corr_matrix


def extract_scene_data(scene_ids, cond, condlist, masked_betas):
    """
    Extract beta patterns and run labels for all scenes in a condition.
    
    Returns:
        scene_data: list of arrays, each (n_trials, n_voxels) for one scene
        scene_run_labels: list of arrays with run labels for each scene
        scene_indices: list of boolean arrays (indices into full beta matrix)
    """
    scene_data = []
    scene_run_labels = []
    scene_indices = []
    
    for s in scene_ids:
        if "IC" in cond:
            indices = condlist[f"conds_spec_{cond}"].str.contains(f"{s}").to_list()
        else:
            indices = condlist['conds_spec'].str.contains(f'{s}').to_list()
        
        indices = np.array(indices)
        scene_data.append(masked_betas[indices])
        scene_run_labels.append(condlist.loc[indices, 'runs'].values)
        scene_indices.append(indices)
    
    return scene_data, scene_run_labels, scene_indices


def compute_train_averages_vectorized(scene_data, scene_run_labels, runs):
    """
    Compute leave-one-run-out training averages for all scenes.
    
    Returns:
        train_avgs: (n_scenes, n_runs, n_voxels) array
                    train_avgs[s, r, :] = mean of scene s data excluding run r
                    NaN if no training data available for that scene/run combination
    """
    n_scenes = len(scene_data)
    n_runs = len(runs)
    n_voxels = scene_data[0].shape[1]
    
    # Initialize with NaN to indicate missing data
    train_avgs = np.full((n_scenes, n_runs, n_voxels), np.nan)
    
    for s_idx, (data, labels) in enumerate(zip(scene_data, scene_run_labels)):
        for r_idx, r in enumerate(runs):
            train_mask = labels != r
            if train_mask.sum() > 0:
                train_avgs[s_idx, r_idx] = np.mean(data[train_mask], axis=0)
            # else: stays NaN (no training data for this fold)
    
    return train_avgs


def get_rsa_matrix_vectorized(conds, condlist, masked_betas):
    """
    Vectorized RSA matrix computation using matrix operations.
    
    Handles both within-condition (e.g., V-V, A-A) and across-condition 
    (e.g., V-A, A-IC_A) comparisons.
    
    For within-condition:
        - Diagonal: reliability (test vs train of same scene)
        - Off-diagonal: similarity between different scenes
        
    For across-condition:
        - All cells: cross-modal similarity
        - Diagonal represents same-scene across modalities
        - Off-diagonal represents different-scene across modalities
    """
    # Create scene IDs
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    
    runs = np.array(sorted(condlist['runs'].unique()))
    n_runs = len(runs)
    n_scenes = 10
    
    # Check if within-condition or across-condition comparison
    same_condition = (conds[0] == conds[1])
    
    # Extract all scene data upfront
    data_1, labels_1, _ = extract_scene_data(scene_id_1, conds[0], condlist, masked_betas)
    
    if same_condition:
        data_2, labels_2 = data_1, labels_1
    else:
        data_2, labels_2, _ = extract_scene_data(scene_id_2, conds[1], condlist, masked_betas)
    
    # Pre-compute training averages for each scene and held-out run
    # Shape: (n_scenes, n_runs, n_voxels)
    train_avgs_1 = compute_train_averages_vectorized(data_1, labels_1, runs)
    
    if same_condition:
        train_avgs_2 = train_avgs_1
    else:
        train_avgs_2 = compute_train_averages_vectorized(data_2, labels_2, runs)
    
    # Initialize RSA matrix with NaN (will store Fisher z-values)
    # NaN indicates no valid data for that cell (e.g., all trials excluded)
    rsa_matrix = np.full((n_scenes, n_scenes), np.nan)
    
    # For each cell in the RSA matrix
    for i in range(n_scenes):
        # For within-condition, only compute upper triangle + diagonal
        j_start = i if same_condition else 0
        
        for j in range(j_start, n_scenes):
            corrs_z = []
            
            # DIAGONAL (same scene) - only for within-condition
            if i == j and same_condition:
                # Within-scene reliability
                data = data_1[i]
                run_labels = labels_1[i]
                
                for r_idx, r in enumerate(runs):
                    test_mask = run_labels == r
                    if test_mask.sum() == 0:
                        continue
                    
                    test_data = data[test_mask]  # (n_test_trials, n_voxels)
                    train_avg = train_avgs_1[i, r_idx]  # (n_voxels,)
                    
                    # Correlate each test trial with train average
                    for test_pattern in test_data:
                        # Use vectorized correlation
                        corr = correlation_matrix(
                            test_pattern.reshape(1, -1), 
                            train_avg.reshape(1, -1)
                        )[0, 0]
                        
                        if not np.isnan(corr):
                            corrs_z.append(fisher_z(corr))
            
            # OFF-DIAGONAL or CROSS-CONDITION
            else:
                for r_idx, r in enumerate(runs):
                    # Direction 1: Scene i test vs Scene j train
                    test_mask_i = labels_1[i] == r
                    if test_mask_i.sum() > 0:
                        test_data_i = data_1[i][test_mask_i]  # (n_test, n_voxels)
                        train_avg_j = train_avgs_2[j, r_idx]  # (n_voxels,)
                        
                        # Vectorized: correlate all test patterns with train average at once
                        corrs = correlation_matrix(
                            test_data_i, 
                            train_avg_j.reshape(1, -1)
                        )[:, 0]  # (n_test,)
                        
                        for corr in corrs:
                            if not np.isnan(corr):
                                corrs_z.append(fisher_z(corr))
                    
                    # Direction 2: Scene j test vs Scene i train
                    test_mask_j = labels_2[j] == r
                    if test_mask_j.sum() > 0:
                        test_data_j = data_2[j][test_mask_j]  # (n_test, n_voxels)
                        train_avg_i = train_avgs_1[i, r_idx]  # (n_voxels,)
                        
                        corrs = correlation_matrix(
                            test_data_j, 
                            train_avg_i.reshape(1, -1)
                        )[:, 0]
                        
                        for corr in corrs:
                            if not np.isnan(corr):
                                corrs_z.append(fisher_z(corr))
            
            # Average Fisher z-values
            if len(corrs_z) > 0:
                mean_z = np.mean(corrs_z)
                rsa_matrix[i, j] = mean_z
                
                # Mirror for within-condition (symmetric matrix)
                if same_condition and i != j:
                    rsa_matrix[j, i] = mean_z
    
    # Convert to DataFrame for compatibility with rest of pipeline
    rsa_corrs = pd.DataFrame(rsa_matrix, index=scene_id_1, columns=scene_id_2)
    
    return rsa_corrs


def get_rsa_matrix_fully_vectorized(conds, condlist, masked_betas):
    """
    Fully vectorized RSA using batch matrix multiplication.
    
    This version computes all correlations for a given run in one matrix operation,
    trading memory for speed. Best for larger ROIs.
    
    Handles both within-condition and across-condition comparisons.
    """
    # Create scene IDs
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    
    runs = np.array(sorted(condlist['runs'].unique()))
    n_runs = len(runs)
    n_scenes = 10
    
    same_condition = (conds[0] == conds[1])
    
    # Extract scene data
    data_1, labels_1, _ = extract_scene_data(scene_id_1, conds[0], condlist, masked_betas)
    
    if same_condition:
        data_2, labels_2 = data_1, labels_1
    else:
        data_2, labels_2, _ = extract_scene_data(scene_id_2, conds[1], condlist, masked_betas)
    
    # Pre-compute training averages
    train_avgs_1 = compute_train_averages_vectorized(data_1, labels_1, runs)
    train_avgs_2 = train_avgs_1 if same_condition else compute_train_averages_vectorized(data_2, labels_2, runs)
    
    # Accumulate Fisher z-values for each cell
    # Using lists of lists to handle variable number of correlations per cell
    z_accumulator = [[[] for _ in range(n_scenes)] for _ in range(n_scenes)]
    
    # Initialize RSA matrix with NaN
    # NaN indicates no valid data for that cell (e.g., all trials excluded)
    rsa_matrix = np.full((n_scenes, n_scenes), np.nan)
    
    # Process each run
    for r_idx, r in enumerate(runs):
        # Get training averages for this fold (n_scenes, n_voxels)
        train_1_fold = train_avgs_1[:, r_idx, :]  # (10, n_voxels)
        train_2_fold = train_avgs_2[:, r_idx, :]  # (10, n_voxels)
        
        # For each scene in condition 1, get test data and correlate with all train averages
        for i in range(n_scenes):
            test_mask = labels_1[i] == r
            if test_mask.sum() == 0:
                continue
            
            test_data = data_1[i][test_mask]  # (n_test, n_voxels)
            
            # Correlate test patterns with ALL condition 2 training averages at once
            # Result: (n_test, n_scenes)
            corr_matrix = correlation_matrix(test_data, train_2_fold)
            
            # Store correlations for each scene pair
            j_start = i if same_condition else 0
            for j in range(j_start, n_scenes):
                corrs = corr_matrix[:, j]
                for corr in corrs:
                    if not np.isnan(corr):
                        z_accumulator[i][j].append(fisher_z(corr))
        
        # For across-condition or off-diagonal: also test condition 2 against condition 1 train
        for j in range(n_scenes):
            test_mask = labels_2[j] == r
            if test_mask.sum() == 0:
                continue
            
            test_data = data_2[j][test_mask]  # (n_test, n_voxels)
            
            # Correlate with ALL condition 1 training averages
            corr_matrix = correlation_matrix(test_data, train_1_fold)
            
            # Store correlations
            i_end = j if same_condition else n_scenes
            for i in range(i_end if same_condition else n_scenes):
                # Skip diagonal for same condition (already handled above)
                if same_condition and i == j:
                    continue
                    
                corrs = corr_matrix[:, i]
                for corr in corrs:
                    if not np.isnan(corr):
                        z_accumulator[i][j].append(fisher_z(corr))
    
    # Average z-values for each cell
    # rsa_matrix already initialized with NaN above
    for i in range(n_scenes):
        j_start = i if same_condition else 0
        for j in range(j_start, n_scenes):
            if len(z_accumulator[i][j]) > 0:
                rsa_matrix[i, j] = np.mean(z_accumulator[i][j])
                if same_condition and i != j:
                    rsa_matrix[j, i] = rsa_matrix[i, j]
            # else: stays NaN (no valid correlations for this cell)
    
    # Convert to DataFrame
    rsa_corrs = pd.DataFrame(rsa_matrix, index=scene_id_1, columns=scene_id_2)
    
    return rsa_corrs


# Keep the original function as a fallback/reference
def get_rsa_matrix_original(conds, condlist, masked_betas):
    """Original loop-based RSA matrix computation (kept for validation)."""
    
    # Now we need to filter, sort, RSA, and visualise
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]

    # Initialise RSA matrix in pandas
    rsa_corrs = pd.DataFrame(index=scene_id_1, columns=scene_id_2)

    # Create a log on comparisons we have done
    log = []
    
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

            if s != ss: # Off-diagonal
                # Get the data
                s1_data = masked_betas[ind1]
                s1_labels = condlist[ind1]
                s2_data = masked_betas[ind2]
                s2_labels = condlist[ind2]

                # List to store Fisher z-values
                corrs_z = []

                # Create folds
                for r in runs:
                    # Get run indices
                    s1_test_idx = (s1_labels['runs'] == r).to_list()
                    s2_test_idx = (s2_labels['runs'] == r).to_list()

                    # Extract the test & train data
                    s1_test_data = s1_data[s1_test_idx]
                    s1_train_data = s1_data[[not x for x in s1_test_idx]]
                    s2_test_data = s2_data[s2_test_idx]
                    s2_train_data = s2_data[[not x for x in s2_test_idx]]

                    # Average the train data
                    s1_train_data = np.mean(s1_train_data, axis=0) 
                    s2_train_data = np.mean(s2_train_data, axis=0) 

                    corr1 = np.corrcoef(s1_test_data, s2_train_data)
                    corr2 = np.corrcoef(s2_test_data, s1_train_data)

                    # Fisher transform before appending
                    if corr1 != []:
                        corrs_z.append(fisher_z(corr1[0, 1]))
                    if corr2 != []:
                        corrs_z.append(fisher_z(corr2[0, 1]))

                # Average all Fisher z-values across folds
                mean_z = np.mean(corrs_z)
                rsa_corrs.loc[s, ss] = mean_z

            else:  # Diagonal
                # Extract the data
                s_data = masked_betas[ind1]
                s_labels = condlist[ind1]

                # Store the Fisher z-values for each split
                corrs_z = []
                for run in runs:
                    # Extract run mask
                    run_idx = (s_labels['runs'] == run).to_list()

                    # Extract test and train data
                    test_data = s_data[run_idx]
                    train_data = s_data[[not x for x in run_idx]]

                    # Average the train data
                    train_data = np.mean(train_data, axis=0)
                    
                    # Correlate the train and test data
                    corr = np.corrcoef(train_data, test_data)
                    
                    # Fisher transform BEFORE appending
                    if corr != []:
                        corrs_z.append(fisher_z(corr[0, 1]))

                # Average the Fisher z-values across runs
                mean_z = np.mean(corrs_z)
                rsa_corrs.loc[s, ss] = mean_z

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)

    return rsa_corrs


# =============================================================================
# SCENE CLASSIFICATION FUNCTIONS (unchanged)
# =============================================================================

def perform_scene_classification(preproc_dir, sub, roi, nruns, cond, hpc, pos, norm, exclusion_path, apply_exclusion):
    """
    Performs scene classification using RSA within a condition and calculates d-prime.
    """
    # Get the GLM Single Betas
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f'mm{sub}' in exclusion_df['subs'].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else: 
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

    # Get the associated labels
    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)

    # Mask the betas
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm)
    
    # Create scene IDs for the condition
    scene_ids = [f"S{i}_{cond}" for i in range(1, 11)]
    
    # Get unique runs
    runs = condlist['runs'].unique()
    
    # Store results for each scene pair comparison
    individual_results = []
    
    # Initialize aggregated statistics
    total_hits = 0
    total_misses = 0
    total_false_alarms = 0
    total_correct_rejections = 0
    
    # For each pair of scenes, perform binary classification
    for i in range(len(scene_ids)):
        for j in range(i+1, len(scene_ids)):
            scene1 = scene_ids[i]
            scene2 = scene_ids[j]
            
            # Get indices for scene 1
            if "IC" in cond:
                s1_indices = condlist[f"conds_spec_{cond}"].str.contains(f"{scene1}").to_list()
            else:
                s1_indices = condlist['conds_spec'].str.contains(f'{scene1}').to_list()
            
            # Get indices for scene 2
            if "IC" in cond:
                s2_indices = condlist[f"conds_spec_{cond}"].str.contains(f"{scene2}").to_list()
            else:
                s2_indices = condlist['conds_spec'].str.contains(f'{scene2}').to_list()
            
            s1_data = masked_betas[s1_indices]
            s1_labels = condlist[s1_indices]
            
            s2_data = masked_betas[s2_indices]
            s2_labels = condlist[s2_indices]
            
            pair_hits = 0
            pair_misses = 0
            pair_false_alarms = 0
            pair_correct_rejections = 0
            
            for r in runs:
                test_idx1 = (s1_labels['runs'] == r).to_list()
                
                if sum(test_idx1) == 0:
                    continue
                
                test_data1 = s1_data[test_idx1]
                
                if len(test_data1) == 0 or np.isnan(test_data1).all():
                    continue
                
                train_idx1 = (s1_labels['runs'] != r).to_list()
                train_data1 = s1_data[train_idx1]
                
                train_idx2 = (s2_labels['runs'] != r).to_list()
                train_data2 = s2_data[train_idx2]
                
                if len(train_data1) == 0 or np.isnan(train_data1).all() or len(train_data2) == 0 or np.isnan(train_data2).all():
                    continue
                
                avg_train_data1 = np.mean(train_data1, axis=0)
                avg_train_data2 = np.mean(train_data2, axis=0)
                
                corr1 = np.corrcoef(test_data1.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data1.flatten(), avg_train_data2.flatten())[0, 1]
                
                if np.isnan(corr1) or np.isnan(corr2):
                    continue
                
                if corr1 > corr2:
                    pair_hits += 1
                else:
                    pair_misses += 1
                
                test_idx2 = (s2_labels['runs'] == r).to_list()
                
                if sum(test_idx2) == 0:
                    continue
                
                test_data2 = s2_data[test_idx2]
                
                if len(test_data2) == 0 or np.isnan(test_data2).all():
                    continue
                
                corr1 = np.corrcoef(test_data2.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data2.flatten(), avg_train_data2.flatten())[0, 1]
                
                if np.isnan(corr1) or np.isnan(corr2):
                    continue
                
                if corr1 > corr2:
                    pair_false_alarms += 1
                else:
                    pair_correct_rejections += 1
            
            total_hits += pair_hits
            total_misses += pair_misses
            total_false_alarms += pair_false_alarms
            total_correct_rejections += pair_correct_rejections
            
            pair_total = pair_hits + pair_misses + pair_false_alarms + pair_correct_rejections
            
            if pair_total == 0:
                continue
                
            pair_accuracy = (pair_hits + pair_correct_rejections) / pair_total
            
            pair_hit_rate = pair_hits / (pair_hits + pair_misses) if (pair_hits + pair_misses) > 0 else 0.5
            pair_fa_rate = pair_false_alarms / (pair_false_alarms + pair_correct_rejections) if (pair_false_alarms + pair_correct_rejections) > 0 else 0.5
            
            pair_hit_rate = max(min(pair_hit_rate, 0.999), 0.001)
            pair_fa_rate = max(min(pair_fa_rate, 0.999), 0.001)
            
            pair_d_prime = stats.norm.ppf(pair_hit_rate) - stats.norm.ppf(pair_fa_rate)
            
            individual_results.append({
                'scene_pair': f"{scene1}_vs_{scene2}",
                'condition': cond,
                'd_prime': pair_d_prime,
                'accuracy': pair_accuracy,
                'hits': pair_hits,
                'misses': pair_misses,
                'false_alarms': pair_false_alarms,
                'correct_rejections': pair_correct_rejections,
                'total_trials': pair_total
            })
    
    total_trials = total_hits + total_misses + total_false_alarms + total_correct_rejections
    
    if total_trials > 0:
        aggregated_accuracy = (total_hits + total_correct_rejections) / total_trials
        
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.5
        fa_rate = total_false_alarms / (total_false_alarms + total_correct_rejections) if (total_false_alarms + total_correct_rejections) > 0 else 0.5
        
        hit_rate = max(min(hit_rate, 0.999), 0.001)
        fa_rate = max(min(fa_rate, 0.999), 0.001)
        
        aggregated_d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    else:
        aggregated_accuracy = np.nan
        aggregated_d_prime = np.nan
    
    aggregated_stats = {
        'condition': cond,
        'hits': total_hits,
        'misses': total_misses,
        'false_alarms': total_false_alarms,
        'correct_rejections': total_correct_rejections,
        'total_trials': total_trials,
        'd_prime': aggregated_d_prime,
        'accuracy': aggregated_accuracy
    }
    
    return aggregated_d_prime, aggregated_accuracy, individual_results, aggregated_stats


def perform_scene_classification_scene_level(preproc_dir, sub, roi, nruns, cond, hpc, pos, norm, exclusion_path, apply_exclusion):
    """Performs scene classification using RSA within a condition and calculates d-prime."""
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f'mm{sub}' in exclusion_df['subs'].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else: 
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm)
    scene_ids = [f"S{i}_{cond}" for i in range(1, 11)]
    runs = condlist['runs'].unique()
    individual_results = []
    
    for i in range(len(scene_ids)):
        for j in range(i+1, len(scene_ids)):
            scene1 = scene_ids[i]
            scene2 = scene_ids[j]
            
            if "IC" in cond:
                s1_indices = condlist[f"conds_spec_{cond}"].str.contains(f"{scene1}").to_list()
            else:
                s1_indices = condlist['conds_spec'].str.contains(f'{scene1}').to_list()
            
            if "IC" in cond:
                s2_indices = condlist[f"conds_spec_{cond}"].str.contains(f"{scene2}").to_list()
            else:
                s2_indices = condlist['conds_spec'].str.contains(f'{scene2}').to_list()
            
            s1_data = masked_betas[s1_indices]
            s1_labels = condlist[s1_indices]
            
            s2_data = masked_betas[s2_indices]
            s2_labels = condlist[s2_indices]
            
            hits = 0
            misses = 0
            false_alarms = 0
            correct_rejections = 0
            
            for r in runs:
                test_idx1 = (s1_labels['runs'] == r).to_list()
                
                if sum(test_idx1) == 0:
                    continue
                
                test_data1 = s1_data[test_idx1]
                
                if len(test_data1) == 0 or np.isnan(test_data1).all():
                    continue
                
                train_idx1 = (s1_labels['runs'] != r).to_list()
                train_data1 = s1_data[train_idx1]
                
                train_idx2 = (s2_labels['runs'] != r).to_list()
                train_data2 = s2_data[train_idx2]
                
                if len(train_data1) == 0 or np.isnan(train_data1).all() or len(train_data2) == 0 or np.isnan(train_data2).all():
                    continue
                
                avg_train_data1 = np.mean(train_data1, axis=0)
                avg_train_data2 = np.mean(train_data2, axis=0)
                
                corr1 = np.corrcoef(test_data1.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data1.flatten(), avg_train_data2.flatten())[0, 1]
                
                if np.isnan(corr1) or np.isnan(corr2):
                    continue
                
                if corr1 > corr2:
                    hits += 1
                else:
                    misses += 1
                
                test_idx2 = (s2_labels['runs'] == r).to_list()
                test_data2 = s2_data[test_idx2]
                
                corr1 = np.corrcoef(test_data2.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data2.flatten(), avg_train_data2.flatten())[0, 1]
                
                if corr1 > corr2:
                    false_alarms += 1
                else:
                    correct_rejections += 1
            
            total = hits + misses + false_alarms + correct_rejections
            
            if total == 0:
                continue
                
            accuracy = (hits + correct_rejections) / total
            
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.5
            fa_rate = false_alarms / (false_alarms + correct_rejections) if (false_alarms + correct_rejections) > 0 else 0.5
            
            hit_rate = max(min(hit_rate, 0.999), 0.001)
            fa_rate = max(min(fa_rate, 0.999), 0.001)
            
            d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
            
            individual_results.append({
                'scene_pair': f"{scene1}_vs_{scene2}",
                'condition': "V",
                'd_prime': d_prime,
                'accuracy': accuracy,
                'hits': hits,
                'misses': misses,
                'false_alarms': false_alarms,
                'correct_rejections': correct_rejections,
                'total_trials': total
            })
    
    if len(individual_results) > 0:
        avg_d_prime = np.mean([r['d_prime'] for r in individual_results])
        avg_accuracy = np.mean([r['accuracy'] for r in individual_results])
    else:
        avg_d_prime = np.nan
        avg_accuracy = np.nan
    
    return avg_d_prime, avg_accuracy, individual_results


def run_scene_classification_analysis(rois, rois_bool, pos, subs, nruns, norm, fisher, user,
                          preproc_dir, conditions, df, results_path, exclusion_path, apply_exclusion):
    """Run scene classification analysis on multiple subjects and ROIs"""
    for sub in subs:
        for cond in conditions:
            t1 = time.time()
            for r in range(len(rois)):
                aggregated_d_prime, aggregated_accuracy, individual_results, aggregated_stats = perform_scene_classification(
                    preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                    cond=cond, hpc=rois_bool[r], pos=pos[r], norm=norm, 
                    exclusion_path=exclusion_path, apply_exclusion=apply_exclusion
                )
                
                if rois[r] == "HPC":
                    roi_name = f'{pos[r]}_{rois[r]}'
                else:
                    roi_name = rois[r]
                
                df.loc[len(df)] = [
                    sub, 
                    cond, 
                    roi_name, 
                    aggregated_d_prime, 
                    aggregated_accuracy,
                    individual_results,
                    len(individual_results),
                    aggregated_stats,
                ]
                
            t2 = time.time()
            print(f"Finished scene classification for {cond} in {sub} in {t2 - t1} seconds", flush=True)

    df['roi'] = df['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": 'TP'
    })

    df.to_csv(results_path, index=False)
    return df


# =============================================================================
# VISUALIZATION AND ANALYSIS FUNCTIONS
# =============================================================================

def visualise_rsa(rsa_corrs, roi):
    plt.figure(figsize=(8, 8))
    sns.heatmap(rsa_corrs, center=0, cmap='coolwarm', annot=False, square=True, cbar=True, xticklabels=True,
                yticklabels=True)
    plt.title(f"RSA Matrix Heatmap {roi}", fontsize=16)
    plt.xlabel("Stimulus ID", fontsize=12)
    plt.ylabel("Stimulus ID", fontsize=12)
    plt.tight_layout()
    plt.show()
    return 0


def run_rsa(preproc_dir, sub, roi, nruns, conds, hpc, vis, pos, norm, exclusion_path, apply_exclusion, atlas, use_vectorized=True):
    """
    Run entire RSA pipeline.
    
    Parameters:
        use_vectorized: bool, if True uses the vectorized matrix multiplication version
    """
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f'mm{sub}' in exclusion_df['subs'].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else: 
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm, atlas)
    
    # Choose RSA function based on parameter
    if use_vectorized:
        rsa_corrs = get_rsa_matrix_fully_vectorized(conds, condlist, masked_betas)
    else:
        rsa_corrs = get_rsa_matrix_original(conds, condlist, masked_betas)

    if vis: visualise_rsa(rsa_corrs, roi)

    return rsa_corrs


def simple_average(rsa_matrix, cond, fisher):
    """Get the right metrics when averaging the entire matrix.
    Uses nanmean to handle excluded trials (NaN values).
    """
    if cond.split('-')[0] != cond.split('-')[1]:
        mean_z = np.nanmean(rsa_matrix)
        t = "with_diag"
    else:
        mean_z = np.nanmean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
        t = "without_diag"
    
    if fisher:
        mean = inverse_fisher_z(mean_z)
        return [mean], [mean_z], [t]
    else:
        return [inverse_fisher_z(mean_z)], [t]


def cross_modal(rsa_matrix, cond, fisher):
    """Get the right metrics when doing cross-modal analysis.
    Uses nanmean to handle excluded trials (NaN values).
    
    For diagonal: if one direction has data but the other is NaN, 
    we keep the valid value (nanmean handles this automatically).
    """
    # Use nanmean to ignore NaN values (from excluded trials)
    diag_values = np.diag(rsa_matrix)
    mean_diag_z = np.nanmean(diag_values)
    
    off_diag_mask = ~np.eye(len(rsa_matrix), dtype=bool)
    mean_off_diag_z = np.nanmean(rsa_matrix[off_diag_mask])
    
    if fisher:
        mean_diag = inverse_fisher_z(mean_diag_z)
        mean_off_diag = inverse_fisher_z(mean_off_diag_z)
        return [mean_diag, mean_off_diag], [mean_diag_z, mean_off_diag_z], ["diag", "off_diag"]
    else:
        return [inverse_fisher_z(mean_diag_z), inverse_fisher_z(mean_off_diag_z)], ["diag", "off_diag"]


def run_rsa_analysis(rois, rois_bool, pos, subs, nruns, norm, fisher, user,
                     vis, preproc_dir, conditions, df, analysis, results_path, 
                     exclusion_path, apply_exclusion, atlas, use_vectorized=True):
    """
    Run RSA analysis on multiple subjects and ROIs.
    
    Parameters:
        use_vectorized: bool, if True uses matrix multiplication version (faster)
    """
    for sub in subs:
        for cond in conditions:
            t1 = time.time()
            for r in range(len(rois)):
                rsa_matrix = run_rsa(preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                                     conds=cond.split('-'), hpc=rois_bool[r], vis=vis, pos=pos[r], 
                                     norm=norm, exclusion_path=exclusion_path, 
                                     apply_exclusion=apply_exclusion, atlas=atlas,
                                     use_vectorized=use_vectorized)
                rsa_matrix = rsa_matrix.to_numpy()

                if fisher:
                    metrics, metrics_z, types = analysis(rsa_matrix, cond, fisher)
                else:
                    metrics, types = analysis(rsa_matrix, cond, fisher)
                    
                if rois[r] == "HPC":
                    roi_name = f'{pos[r]}_{rois[r]}'
                else:
                    roi_name = rois[r]
                    
                if fisher:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], metrics_z[m], types[m]]
    
                else:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], types[m]]
            
            t2 = time.time()
            print(f"Finished all ROIs for {cond} in {sub} in {t2 - t1} seconds", flush=True)

    df['roi'] = df['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": 'TP'
    })

    df.head(10)
    df.to_csv(results_path, index=False)
    return df


def visualise_rsa_analysis(df, sensory, analysis, hue_type, bars=False):
    df_vis = df
    if sensory:
        df_vis = df_vis[~df_vis.roi.isin(["CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum"])]
    else:
        df_vis = df_vis[df_vis.roi.isin(["HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum"])]

    sns.set_style("whitegrid")
    sns.color_palette("Paired")

    ax = sns.catplot(x='roi', y='corr', hue=hue_type, data=df_vis, kind='bar', height=5, aspect=2, palette='Paired')
    bb = sns.stripplot(x='roi', y='corr', hue=hue_type, data=df_vis, color="grey", dodge=True)

    xticks = {roi: i for i, roi in enumerate(df_vis['roi'].unique())}

    if bars:
        for roi in df_vis['roi'].unique():
            sub_df = df_vis[df_vis['roi'] == roi]

            for sub in sub_df['sub'].unique():
                sub_data = sub_df[sub_df['sub'] == sub]

                x_pos = xticks[roi]
                x_vals = [x_pos - 0.2, x_pos + 0.2]
                y_vals = sub_data['corr'].values

                line_color = 'green' if y_vals[0] > y_vals[1] else 'red'

                ax.ax.plot(x_vals, y_vals, color=line_color, linewidth=0.8, alpha=0.7)

    ax.set_axis_labels("ROI", "R")
    ax.despine(left=True)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels)
    ax.ax.set_title(f"Metrics for {'Sensory' if sensory else 'HPC'} ROIs with {analysis.__name__} analysis")

    return "Plotted successfully"


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_vectorized_vs_original(preproc_dir, sub, roi, nruns, conds, hpc, pos, norm, 
                                     exclusion_path, apply_exclusion, atlas, rtol=1e-5):
    """
    Compare vectorized and original RSA implementations.
    Returns True if results match within tolerance.
    """
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f'mm{sub}' in exclusion_df['subs'].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else: 
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm, atlas)
    
    # Run both versions
    t1 = time.time()
    rsa_original = get_rsa_matrix_original(conds, condlist, masked_betas)
    t2 = time.time()
    rsa_vectorized = get_rsa_matrix_fully_vectorized(conds, condlist, masked_betas)
    t3 = time.time()
    
    # Compare
    orig_arr = rsa_original.to_numpy()
    vec_arr = rsa_vectorized.to_numpy()
    
    # Count NaN cells
    orig_nans = np.sum(np.isnan(orig_arr))
    vec_nans = np.sum(np.isnan(vec_arr))
    
    matches = np.allclose(orig_arr, vec_arr, rtol=rtol, equal_nan=True)
    max_diff = np.nanmax(np.abs(orig_arr - vec_arr))
    
    print(f"Condition: {conds[0]}-{conds[1]}")
    print(f"  Original time: {t2-t1:.3f}s")
    print(f"  Matmul time: {t3-t2:.3f}s")
    print(f"  Speedup: {(t2-t1)/(t3-t2):.1f}x")
    print(f"  Results match: {matches}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  NaN cells - Original: {orig_nans}, Matmul: {vec_nans}")
    
    return matches, rsa_original, rsa_vectorized


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    parser.add_argument('--use_original', action='store_true', help='Use original loop-based implementation')
    p = parser.parse_args()
    
    subs = [p.sub_id]
    atlas = "ASHS"
    
    # ==========================================================================
    # USER SETTINGS - modify these as needed
    # ==========================================================================
    execute = True                      # Run the main RSA analysis
    validation = True                   # Run validation comparing original vs matmul version
    use_vectorized = not p.use_original # Use matrix multiplication version (faster)
    # ==========================================================================

    if atlas == "ASHS":
        rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1", "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1", "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1", "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",  "Superior Temporal Gyrus, posterior division", "Lateral Occipital Cortex, inferior division"]
        rois_bool = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        pos = ["", "", "", "", "", "", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "left", "right", "", "","",""]

    nruns = 9
    norm = False
    fisher = True
    apply_exclusion = True
    user = "or62"
    vis = False
    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed"
    conditions = ['V-V', 'A-A', 'C-C', 'V-IC_V', 'A-IC_A', 'V-A']
    conditions_classification = ['V', 'A', 'C', 'IC_V', 'IC_A']

    df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'z_similarity', 'type'])
    df_classification = pd.DataFrame(columns=['sub', 'cond', 'roi', 'd_prime', 'accuracy', 'individual results','num_scene_pairs', 'aggregated_stats'])

    analysis = cross_modal

    results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/{atlas}/"
    results_file_name = f"sub_{subs[0]}_atlas_{atlas}_fisher_{fisher}_norm_{norm}_exclusion_{apply_exclusion}_RSA_Results.csv"
    
    os.makedirs(results_path, exist_ok=True)
    
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'

    # Run validation to compare vectorized vs original implementation
    if validation:
        print("=" * 60)
        print("VALIDATION: Comparing matrix multiplication vs original loop implementation")
        print("=" * 60)
        # Validate ALL conditions that will be used in the analysis
        all_valid = True
        for cond in conditions:
            matches, _, _ = validate_vectorized_vs_original(
                preproc_dir=preproc_dir, 
                sub=subs[0], 
                roi=rois[6],  # HPC
                nruns=nruns,
                conds=cond.split('-'), 
                hpc=rois_bool[6], 
                pos=pos[6], 
                norm=norm, 
                exclusion_path=exclusion_path, 
                apply_exclusion=apply_exclusion, 
                atlas=atlas
            )
            if not matches:
                all_valid = False
            print()
        
        if all_valid:
            print("✓ All validations passed! Matrix multiplication implementation matches original.")
        else:
            print("✗ WARNING: Some validations failed! Check results carefully.")
            if use_vectorized:
                print("  Consider setting use_vectorized = False to use loop-based implementation.")
        print("=" * 60)
        print()
    else:
        print("Skipping validation (validation = False)")
        print()
    
    # Run main analysis
    if execute: 
        print(f"Running RSA analysis with {'matrix multiplication' if use_vectorized else 'original loop'} implementation...")
        df = run_rsa_analysis(
            rois=rois, rois_bool=rois_bool, pos=pos, subs=subs,
            nruns=nruns, norm=norm, fisher=fisher, user=user,
            vis=vis, preproc_dir=preproc_dir, conditions=conditions,
            df=df, analysis=analysis, results_path=results_path+results_file_name, 
            exclusion_path=exclusion_path, apply_exclusion=apply_exclusion, atlas=atlas,
            use_vectorized=use_vectorized
        )