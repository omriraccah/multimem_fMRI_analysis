#!/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem/bin/python
import warnings
import sys

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import nibabel as nib
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
from nilearn.image import resample_to_img
from os.path import join
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


# =============================================================================
# MATRIX MULTIPLICATION RSA FUNCTIONS (Optimized for Searchlight)
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
    X_z = zscore_rows(X)
    Y_z = zscore_rows(Y)
    n_voxels = X.shape[1]
    return np.dot(X_z, Y_z.T) / n_voxels


def extract_scene_data(scene_ids, cond, condlist, masked_betas):
    """
    Extract beta patterns and run labels for all scenes in a condition.
    
    Returns:
        scene_data: list of arrays, each (n_trials, n_voxels) for one scene
        scene_run_labels: list of arrays with run labels for each scene
    """
    scene_data = []
    scene_run_labels = []
    
    # Pre-compute the column to use for indexing
    if "IC" in cond:
        col = condlist[f"conds_spec_{cond}"]
    else:
        col = condlist['conds_spec']
    
    runs_col = condlist['runs'].values
    
    for s in scene_ids:
        indices = col.str.contains(s).values
        scene_data.append(masked_betas[indices])
        scene_run_labels.append(runs_col[indices])
    
    return scene_data, scene_run_labels


def compute_train_averages(scene_data, scene_run_labels, runs):
    """
    Compute leave-one-run-out training averages for all scenes.
    
    Returns:
        train_avgs: (n_scenes, n_runs, n_voxels) array
    """
    n_scenes = len(scene_data)
    n_runs = len(runs)
    n_voxels = scene_data[0].shape[1]
    
    train_avgs = np.full((n_scenes, n_runs, n_voxels), np.nan)
    
    for s_idx in range(n_scenes):
        data = scene_data[s_idx]
        labels = scene_run_labels[s_idx]
        for r_idx, r in enumerate(runs):
            train_mask = labels != r
            if train_mask.sum() > 0:
                train_avgs[s_idx, r_idx] = data[train_mask].mean(axis=0)
    
    return train_avgs


def get_rsa_matrix(conds, condlist, masked_betas):
    """
    Compute RSA matrix using batch matrix multiplication.
    Optimized for searchlight analysis.
    """
    n_scenes = 10
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    
    runs = np.array(sorted(condlist['runs'].unique()))
    n_runs = len(runs)
    
    same_condition = (conds[0] == conds[1])
    
    # Extract scene data
    data_1, labels_1 = extract_scene_data(scene_id_1, conds[0], condlist, masked_betas)
    
    if same_condition:
        data_2, labels_2 = data_1, labels_1
    else:
        data_2, labels_2 = extract_scene_data(scene_id_2, conds[1], condlist, masked_betas)
    
    # Pre-compute training averages: (n_scenes, n_runs, n_voxels)
    train_avgs_1 = compute_train_averages(data_1, labels_1, runs)
    train_avgs_2 = train_avgs_1 if same_condition else compute_train_averages(data_2, labels_2, runs)
    
    # Pre-allocate arrays to collect z-values (estimate max size)
    # Max correlations per cell ≈ n_runs * 2 (both directions)
    max_corrs_per_cell = n_runs * 2
    z_values = np.full((n_scenes, n_scenes, max_corrs_per_cell), np.nan)
    z_counts = np.zeros((n_scenes, n_scenes), dtype=int)
    
    # Process each run
    for r_idx, r in enumerate(runs):
        train_1_fold = train_avgs_1[:, r_idx, :]  # (n_scenes, n_voxels)
        train_2_fold = train_avgs_2[:, r_idx, :]
        
        # Direction 1: condition 1 test vs condition 2 train
        for i in range(n_scenes):
            test_mask = labels_1[i] == r
            n_test = test_mask.sum()
            if n_test == 0:
                continue
            
            test_data = data_1[i][test_mask]
            
            # Compute all correlations at once: (n_test, n_scenes)
            corr_matrix = correlation_matrix(test_data, train_2_fold)
            
            # Fisher z-transform entire matrix at once
            # Clip to avoid infinity at r=1 or r=-1
            corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)
            z_matrix = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))
            
            # Store z-values
            j_start = i if same_condition else 0
            for j in range(j_start, n_scenes):
                z_col = z_matrix[:, j]
                valid = ~np.isnan(z_col)
                n_valid = valid.sum()
                if n_valid > 0:
                    start_idx = z_counts[i, j]
                    z_values[i, j, start_idx:start_idx + n_valid] = z_col[valid]
                    z_counts[i, j] += n_valid
        
        # Direction 2: condition 2 test vs condition 1 train
        for j in range(n_scenes):
            test_mask = labels_2[j] == r
            n_test = test_mask.sum()
            if n_test == 0:
                continue
            
            test_data = data_2[j][test_mask]
            
            corr_matrix = correlation_matrix(test_data, train_1_fold)
            corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)
            z_matrix = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))
            
            i_end = j if same_condition else n_scenes
            for i in range(i_end if same_condition else n_scenes):
                if same_condition and i == j:
                    continue
                
                z_col = z_matrix[:, i]
                valid = ~np.isnan(z_col)
                n_valid = valid.sum()
                if n_valid > 0:
                    start_idx = z_counts[i, j]
                    z_values[i, j, start_idx:start_idx + n_valid] = z_col[valid]
                    z_counts[i, j] += n_valid
    
    # Compute mean z-values using nanmean on pre-allocated array
    rsa_matrix = np.full((n_scenes, n_scenes), np.nan)
    
    for i in range(n_scenes):
        j_start = i if same_condition else 0
        for j in range(j_start, n_scenes):
            if z_counts[i, j] > 0:
                rsa_matrix[i, j] = np.mean(z_values[i, j, :z_counts[i, j]])
                if same_condition and i != j:
                    rsa_matrix[j, i] = rsa_matrix[i, j]
    
    # Convert to DataFrame
    rsa_corrs = pd.DataFrame(rsa_matrix, index=scene_id_1, columns=scene_id_2)
    
    return rsa_corrs


def get_rsa_matrix_fast(conds, condlist, masked_betas, verbose=False):
    """
    Fastest RSA matrix computation - returns just the numpy array.
    Use this for searchlight where you don't need DataFrame output.
    
    Parameters:
        verbose: if True, prints diagnostic info about exclusions
    
    Returns:
        rsa_matrix: (10, 10) numpy array of Fisher z-values
        exclusion_info: dict with exclusion diagnostics (only if verbose=True)
    """
    n_scenes = 10
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    
    runs = np.array(sorted(condlist['runs'].unique()))
    n_runs = len(runs)
    
    same_condition = (conds[0] == conds[1])
    
    # Extract scene data
    data_1, labels_1 = extract_scene_data(scene_id_1, conds[0], condlist, masked_betas)
    
    if same_condition:
        data_2, labels_2 = data_1, labels_1
    else:
        data_2, labels_2 = extract_scene_data(scene_id_2, conds[1], condlist, masked_betas)
    
    # Pre-compute training averages
    train_avgs_1 = compute_train_averages(data_1, labels_1, runs)
    train_avgs_2 = train_avgs_1 if same_condition else compute_train_averages(data_2, labels_2, runs)
    
    # Track exclusion info if verbose
    if verbose:
        exclusion_info = {
            'condition': f"{conds[0]}-{conds[1]}",
            'n_runs': n_runs,
            'runs': runs.tolist(),
            'trials_per_scene_cond1': {},
            'trials_per_scene_cond2': {},
            'missing_test_data': [],  # (scene, run) pairs with no test data
            'missing_train_data': [],  # (scene, run) pairs with no train data
            'correlations_per_cell': np.zeros((n_scenes, n_scenes), dtype=int),
            'expected_correlations_per_cell': n_runs * 2 if not same_condition else n_runs * 2,
        }
        
        # Count trials per scene
        for i, scene in enumerate(scene_id_1):
            exclusion_info['trials_per_scene_cond1'][scene] = len(data_1[i])
            trials_by_run = {r: (labels_1[i] == r).sum() for r in runs}
            exclusion_info['trials_per_scene_cond1'][f"{scene}_by_run"] = trials_by_run
        
        if not same_condition:
            for j, scene in enumerate(scene_id_2):
                exclusion_info['trials_per_scene_cond2'][scene] = len(data_2[j])
                trials_by_run = {r: (labels_2[j] == r).sum() for r in runs}
                exclusion_info['trials_per_scene_cond2'][f"{scene}_by_run"] = trials_by_run
        
        # Check for missing training data (NaN in train_avgs)
        for i in range(n_scenes):
            for r_idx, r in enumerate(runs):
                if np.isnan(train_avgs_1[i, r_idx, 0]):
                    exclusion_info['missing_train_data'].append((scene_id_1[i], int(r), 'cond1'))
        
        if not same_condition:
            for j in range(n_scenes):
                for r_idx, r in enumerate(runs):
                    if np.isnan(train_avgs_2[j, r_idx, 0]):
                        exclusion_info['missing_train_data'].append((scene_id_2[j], int(r), 'cond2'))
    
    # Accumulate sum and count for online mean calculation
    z_sum = np.zeros((n_scenes, n_scenes))
    z_count = np.zeros((n_scenes, n_scenes), dtype=int)
    
    for r_idx, r in enumerate(runs):
        train_1_fold = train_avgs_1[:, r_idx, :]
        train_2_fold = train_avgs_2[:, r_idx, :]
        
        # Direction 1
        for i in range(n_scenes):
            test_mask = labels_1[i] == r
            if not test_mask.any():
                if verbose:
                    exclusion_info['missing_test_data'].append((scene_id_1[i], int(r), 'cond1_test'))
                continue
            
            test_data = data_1[i][test_mask]
            corr_matrix = correlation_matrix(test_data, train_2_fold)
            corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)
            z_matrix = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))
            
            j_start = i if same_condition else 0
            for j in range(j_start, n_scenes):
                z_col = z_matrix[:, j]
                valid = ~np.isnan(z_col)
                if valid.any():
                    z_sum[i, j] += z_col[valid].sum()
                    z_count[i, j] += valid.sum()
        
        # Direction 2
        for j in range(n_scenes):
            test_mask = labels_2[j] == r
            if not test_mask.any():
                if verbose and not same_condition:
                    exclusion_info['missing_test_data'].append((scene_id_2[j], int(r), 'cond2_test'))
                continue
            
            test_data = data_2[j][test_mask]
            corr_matrix = correlation_matrix(test_data, train_1_fold)
            corr_matrix = np.clip(corr_matrix, -0.9999, 0.9999)
            z_matrix = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))
            
            i_end = j if same_condition else n_scenes
            for i in range(i_end if same_condition else n_scenes):
                if same_condition and i == j:
                    continue
                
                z_col = z_matrix[:, i]
                valid = ~np.isnan(z_col)
                if valid.any():
                    z_sum[i, j] += z_col[valid].sum()
                    z_count[i, j] += valid.sum()
    
    # Compute means
    rsa_matrix = np.full((n_scenes, n_scenes), np.nan)
    valid_cells = z_count > 0
    rsa_matrix[valid_cells] = z_sum[valid_cells] / z_count[valid_cells]
    
    # Mirror for within-condition
    if same_condition:
        for i in range(n_scenes):
            for j in range(i + 1, n_scenes):
                rsa_matrix[j, i] = rsa_matrix[i, j]
    
    if verbose:
        exclusion_info['correlations_per_cell'] = z_count.copy()
        # Also copy for symmetric cells in within-condition
        if same_condition:
            for i in range(n_scenes):
                for j in range(i + 1, n_scenes):
                    exclusion_info['correlations_per_cell'][j, i] = z_count[i, j]
        
        exclusion_info['nan_cells'] = np.sum(np.isnan(rsa_matrix))
        exclusion_info['total_cells'] = n_scenes * n_scenes
        
        return rsa_matrix, exclusion_info
    
    return rsa_matrix


def print_exclusion_report(exclusion_info):
    """Pretty print the exclusion diagnostics."""
    print("=" * 70)
    print(f"EXCLUSION REPORT: {exclusion_info['condition']}")
    print("=" * 70)
    
    print(f"\nRuns included: {exclusion_info['runs']} ({exclusion_info['n_runs']} total)")
    
    print(f"\n--- Trials per scene (Condition 1: {exclusion_info['condition'].split('-')[0]}) ---")
    for key, val in exclusion_info['trials_per_scene_cond1'].items():
        if '_by_run' not in key:
            by_run = exclusion_info['trials_per_scene_cond1'].get(f"{key}_by_run", {})
            run_str = ", ".join([f"R{r}:{c}" for r, c in by_run.items()])
            # Flag if any run has 0 trials
            missing_runs = [r for r, c in by_run.items() if c == 0]
            flag = " *** EXCLUDED IN RUN(S): " + ",".join(map(str, missing_runs)) if missing_runs else ""
            print(f"  {key}: {val} trials  [{run_str}]{flag}")
    
    if exclusion_info['trials_per_scene_cond2']:
        print(f"\n--- Trials per scene (Condition 2: {exclusion_info['condition'].split('-')[1]}) ---")
        for key, val in exclusion_info['trials_per_scene_cond2'].items():
            if '_by_run' not in key:
                by_run = exclusion_info['trials_per_scene_cond2'].get(f"{key}_by_run", {})
                run_str = ", ".join([f"R{r}:{c}" for r, c in by_run.items()])
                missing_runs = [r for r, c in by_run.items() if c == 0]
                flag = " *** EXCLUDED IN RUN(S): " + ",".join(map(str, missing_runs)) if missing_runs else ""
                print(f"  {key}: {val} trials  [{run_str}]{flag}")
    
    if exclusion_info['missing_test_data']:
        print(f"\n--- Missing TEST data (excluded trials) ---")
        for scene, run, direction in exclusion_info['missing_test_data']:
            print(f"  {scene} in Run {run} ({direction})")
    else:
        print(f"\n--- No missing test data ---")
    
    if exclusion_info['missing_train_data']:
        print(f"\n--- Missing TRAIN data (affects leave-one-out average) ---")
        for scene, run, direction in exclusion_info['missing_train_data']:
            print(f"  {scene} when holding out Run {run} ({direction})")
    else:
        print(f"\n--- No missing train data ---")
    
    print(f"\n--- Correlation counts per cell ---")
    corr_counts = exclusion_info['correlations_per_cell']
    expected = exclusion_info['expected_correlations_per_cell']
    
    # Find cells with fewer correlations than expected
    problem_cells = []
    for i in range(corr_counts.shape[0]):
        for j in range(corr_counts.shape[1]):
            if corr_counts[i, j] < expected and corr_counts[i, j] > 0:
                problem_cells.append((i, j, corr_counts[i, j]))
            elif corr_counts[i, j] == 0:
                problem_cells.append((i, j, 0))
    
    if problem_cells:
        print(f"  Cells with fewer than expected ({expected}) correlations:")
        for i, j, count in problem_cells[:20]:  # Limit output
            scene_i = f"S{i+1}"
            scene_j = f"S{j+1}"
            print(f"    [{scene_i}, {scene_j}]: {count} correlations (missing {expected - count})")
        if len(problem_cells) > 20:
            print(f"    ... and {len(problem_cells) - 20} more")
    else:
        print(f"  All cells have expected number of correlations ({expected})")
    
    print(f"\n--- Summary ---")
    print(f"  Total NaN cells in RSA matrix: {exclusion_info['nan_cells']} / {exclusion_info['total_cells']}")
    print("=" * 70)


def get_excluded_trials_from_condlist(condlist):
    """
    Extract which trials were marked as excluded in condlist.
    Returns a summary of exclusions by run.
    """
    excluded = {}
    
    # Find all rows with 'exclude' in conds_spec
    exclude_mask = condlist['conds_spec'].str.contains('exclude', na=False)
    
    if not exclude_mask.any():
        return None
    
    excluded_rows = condlist[exclude_mask]
    
    for run in sorted(condlist['runs'].unique()):
        run_excluded = excluded_rows[excluded_rows['runs'] == run]
        if len(run_excluded) > 0:
            excluded[run] = run_excluded['conds_spec'].tolist()
    
    return excluded


def print_condlist_exclusion_summary(condlist):
    """Print a summary of what was excluded based on condlist."""
    print("=" * 70)
    print("CONDLIST EXCLUSION SUMMARY")
    print("=" * 70)
    
    # Count trials per condition per run
    print("\n--- Trials per condition per run ---")
    for cond in ['V', 'A', 'C', 'IC']:
        cond_mask = condlist['conds_gen'] == cond
        print(f"\n  {cond}:")
        for run in sorted(condlist['runs'].unique()):
            run_mask = condlist['runs'] == run
            count = (cond_mask & run_mask).sum()
            print(f"    Run {run}: {count} trials")
    
    # Show excluded trials
    excluded = get_excluded_trials_from_condlist(condlist)
    
    print("\n--- Excluded trials ---")
    if excluded:
        for run, trials in excluded.items():
            print(f"  Run {run}: {trials}")
    else:
        print("  No excluded trials found")
    
    # Count total valid vs excluded
    total = len(condlist)
    n_excluded = condlist['conds_spec'].str.contains('exclude', na=False).sum()
    print(f"\n--- Totals ---")
    print(f"  Total trials: {total}")
    print(f"  Excluded: {n_excluded}")
    print(f"  Valid: {total - n_excluded}")
    print("=" * 70)


def get_rsa_metrics_fast(conds, condlist, masked_betas):
    """
    Fastest version for searchlight - computes RSA and returns diagonal/off-diagonal means directly.
    Skips DataFrame creation entirely.
    
    Returns:
        mean_diag_z: mean Fisher z of diagonal (same-scene similarity)
        mean_offdiag_z: mean Fisher z of off-diagonal (different-scene similarity)
    """
    rsa_matrix = get_rsa_matrix_fast(conds, condlist, masked_betas)
    
    n = rsa_matrix.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    
    mean_diag_z = np.nanmean(rsa_matrix[diag_mask])
    mean_offdiag_z = np.nanmean(rsa_matrix[~diag_mask])
    
    return mean_diag_z, mean_offdiag_z


def run_rsa(preproc_dir, sub, roi, nruns, conds, hpc, vis, pos, norm, exclusion_path, apply_exclusion, atlas, timing=False, check_exclusions=False):
    """Run entire RSA pipeline.
    
    Parameters:
        timing: if True, prints timing for each step
        check_exclusions: if True, prints detailed exclusion diagnostics
    """
    if timing:
        t_start = time.time()
    
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f'mm{sub}' in exclusion_df['subs'].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else: 
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    
    if timing:
        t_betas = time.time()
        print(f"  [TIMING] load_glm_single: {t_betas - t_start:.3f}s")

    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
    
    if timing:
        t_labels = time.time()
        print(f"  [TIMING] get_labels: {t_labels - t_betas:.3f}s")
    
    # Print condlist exclusion summary if requested
    if check_exclusions:
        print(f"\n>>> Checking exclusions for Subject {sub}, ROI {roi}, Condition {conds[0]}-{conds[1]}")
        print_condlist_exclusion_summary(condlist)
    
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm, atlas)
    
    if timing:
        t_extract = time.time()
        print(f"  [TIMING] extract_betas: {t_extract - t_labels:.3f}s (ROI: {roi}, shape: {masked_betas.shape})")
    
    # Use fast version with verbose output if checking exclusions
    if check_exclusions:
        rsa_matrix, exclusion_info = get_rsa_matrix_fast(conds, condlist, masked_betas, verbose=True)
        print_exclusion_report(exclusion_info)
    else:
        rsa_matrix = get_rsa_matrix_fast(conds, condlist, masked_betas, verbose=False)
    
    # Convert to DataFrame for compatibility
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    rsa_corrs = pd.DataFrame(rsa_matrix, index=scene_id_1, columns=scene_id_2)

    if timing:
        t_rsa = time.time()
        print(f"  [TIMING] get_rsa_matrix_fast: {t_rsa - t_extract:.3f}s")
        print(f"  [TIMING] TOTAL: {t_rsa - t_start:.3f}s")

    if vis: 
        visualise_rsa(rsa_corrs, roi)

    return rsa_corrs


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

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
                     exclusion_path, apply_exclusion, atlas, timing=False, check_exclusions=False):
    """Run RSA analysis on multiple subjects and ROIs.
    
    Parameters:
        timing: if True, prints detailed timing for each step (useful for debugging slowness)
        check_exclusions: if True, prints detailed exclusion diagnostics for FIRST ROI only
    """
    for sub in subs:
        for cond in conditions:
            t1 = time.time()
            for r in range(len(rois)):
                if timing:
                    print(f"\n[TIMING] Sub {sub}, Cond {cond}, ROI {rois[r]}")
                
                # Only check exclusions on first ROI to avoid flooding output
                do_check = check_exclusions and (r == 0)
                
                rsa_matrix = run_rsa(preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                                     conds=cond.split('-'), hpc=rois_bool[r], vis=vis, pos=pos[r], 
                                     norm=norm, exclusion_path=exclusion_path, 
                                     apply_exclusion=apply_exclusion, atlas=atlas, 
                                     timing=timing, check_exclusions=do_check)
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


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualise_rsa(rsa_corrs, roi):
    plt.figure(figsize=(8, 8))
    sns.heatmap(rsa_corrs, center=0, cmap='coolwarm', annot=False, square=True, cbar=True, 
                xticklabels=True, yticklabels=True)
    plt.title(f"RSA Matrix Heatmap {roi}", fontsize=16)
    plt.xlabel("Stimulus ID", fontsize=12)
    plt.ylabel("Stimulus ID", fontsize=12)
    plt.tight_layout()
    plt.show()
    return 0


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
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    p = parser.parse_args()
    
    subs = [p.sub_id]
    atlas = "ASHS"
    
    # ==========================================================================
    # USER SETTINGS - modify these as needed
    # ==========================================================================
    execute = True            # Run the main RSA analysis
    timing = True            # Print detailed timing for each step (set True to debug slowness)
    check_exclusions = True  # Print detailed exclusion diagnostics (set True to verify exclusions)
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

    df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'z_similarity', 'type'])

    analysis = cross_modal

    results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/{atlas}/"
    results_file_name = f"sub_{subs[0]}_atlas_{atlas}_fisher_{fisher}_norm_{norm}_exclusion_{apply_exclusion}_RSA_Results_MATRIXOPT2.csv"
    
    os.makedirs(results_path, exist_ok=True)
    
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'

    # Run main analysis
    if execute: 
        print(f"Running RSA analysis with matrix multiplication implementation...")
        df = run_rsa_analysis(
            rois=rois, rois_bool=rois_bool, pos=pos, subs=subs,
            nruns=nruns, norm=norm, fisher=fisher, user=user,
            vis=vis, preproc_dir=preproc_dir, conditions=conditions,
            df=df, analysis=analysis, results_path=results_path+results_file_name, 
            exclusion_path=exclusion_path, apply_exclusion=apply_exclusion, atlas=atlas,
            timing=timing, check_exclusions=check_exclusions
        )