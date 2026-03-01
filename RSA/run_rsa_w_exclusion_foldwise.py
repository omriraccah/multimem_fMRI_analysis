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
import pickle

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
        # Always use ASHS folder for standard hippocampal ROIs
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

# Function to make the RSA matrix
def get_rsa_matrix(conds, condlist, masked_betas):
    
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

            if s != ss:
                # Get the data
                s1_data = masked_betas[ind1]
                s1_labels = condlist[ind1]
                s2_data = masked_betas[ind2]
                s2_labels = condlist[ind2]

                # List to store correlations
                corrs = []

                # Create folds
                for r in runs:
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
                        s1_train_data = np.mean(s1_train_data, axis=0) 
                        s2_train_data = np.mean(s2_train_data, axis=0) 
                        
                        corr1 = np.corrcoef(s1_test_data, s2_train_data)
                        corr2 = np.corrcoef(s2_test_data, s1_train_data)
                        
                        # Save the results
                        if corr1 != []:
                            corrs.append(corr1[0, 1])
                        if corr2 != []:
                            corrs.append(corr2[0, 1])

                # Average all correlations across folds
                mean_corr = np.mean(corrs)

                # Calculate correlation and store it
                rsa_corrs.loc[s, ss] = mean_corr

            else:  # Diagonal
                # Extract the data
                s_data = masked_betas[ind1]
                s_labels = condlist[ind1]

                # Store the correlations for each split
                corrs = []
                # Perform reliability
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
                    
                    if corr != []:
                        corrs.append(corr[0, 1])

                # Average the reliability across runs
                mean_corr = np.mean(corrs)

                # Store the reliability measure
                rsa_corrs.loc[s, ss] = mean_corr

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)

    return rsa_corrs

# ============================================================================
# NEW FUNCTION: Compute RSA matrices separately for each fold
# ============================================================================
def get_rsa_matrix_by_fold(conds, condlist, masked_betas):
    """
    Compute RSA matrices separately for each fold (left-out run).
    
    Returns:
    - fold_matrices: List of RSA matrices, one per fold
    - fold_runs: List of run numbers corresponding to each fold
    """
    # Get unique runs
    runs = condlist['runs'].unique()
    
    # Initialize storage
    fold_matrices = []
    fold_runs = []
    
    # Create scene IDs
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]
    
    # Loop through each fold (each left-out run)
    for test_run in runs:
        # Initialize RSA matrix for this fold
        rsa_fold = pd.DataFrame(index=scene_id_1, columns=scene_id_2, dtype=float)
        
        for s in scene_id_1:
            for ss in scene_id_2:
                # Get indices
                if "IC" in conds[0]:
                    ind1 = condlist[f"conds_spec_{conds[0]}"].str.contains(f"{s}").to_list()
                else:
                    ind1 = condlist['conds_spec'].str.contains(f'{s}').to_list()
                
                if "IC" in conds[1]:
                    ind2 = condlist[f"conds_spec_{conds[1]}"].str.contains(f"{ss}").to_list()
                else:
                    ind2 = condlist['conds_spec'].str.contains(f'{ss}').to_list()
                
                if s != ss:  # Off-diagonal
                    # Get data
                    s1_data = masked_betas[ind1]
                    s1_labels = condlist[ind1]
                    s2_data = masked_betas[ind2]
                    s2_labels = condlist[ind2]
                    
                    # Test on held-out run
                    s1_test_idx = (s1_labels['runs'] == test_run).to_list()
                    s2_test_idx = (s2_labels['runs'] == test_run).to_list()
                    
                    # Train on all other runs
                    s1_train_idx = (s1_labels['runs'] != test_run).to_list()
                    s2_train_idx = (s2_labels['runs'] != test_run).to_list()
                    
                    # Extract data
                    s1_test_data = s1_data[s1_test_idx]
                    s1_train_data = s1_data[s1_train_idx]
                    s2_test_data = s2_data[s2_test_idx]
                    s2_train_data = s2_data[s2_train_idx]
                    
                    # Average training data
                    s1_train_mean = np.mean(s1_train_data, axis=0)
                    s2_train_mean = np.mean(s2_train_data, axis=0)
                    
                    # Compute correlations (bidirectional)
                    corrs = []
                    if len(s1_test_data) > 0:
                        corr1 = np.corrcoef(s1_test_data, s2_train_mean)
                        if corr1.size > 0:
                            corrs.append(corr1[0, 1])
                    
                    if len(s2_test_data) > 0:
                        corr2 = np.corrcoef(s2_test_data, s1_train_mean)
                        if corr2.size > 0:
                            corrs.append(corr2[0, 1])
                    
                    # Average the two directions
                    rsa_fold.loc[s, ss] = np.mean(corrs) if corrs else np.nan
                    
                else:  # Diagonal (reliability)
                    s_data = masked_betas[ind1]
                    s_labels = condlist[ind1]
                    
                    # Test on held-out run
                    test_idx = (s_labels['runs'] == test_run).to_list()
                    train_idx = (s_labels['runs'] != test_run).to_list()
                    
                    test_data = s_data[test_idx]
                    train_data = s_data[train_idx]
                    
                    # Average training data
                    train_mean = np.mean(train_data, axis=0)
                    
                    # Compute correlation
                    if len(test_data) > 0:
                        corr = np.corrcoef(train_mean, test_data)
                        rsa_fold.loc[s, ss] = corr[0, 1] if corr.size > 0 else np.nan
                    else:
                        rsa_fold.loc[s, ss] = np.nan
        
        fold_matrices.append(rsa_fold.astype(float).to_numpy())
        fold_runs.append(test_run)
    
    return fold_matrices, fold_runs


# ============================================================================
# NEW FUNCTION: Analyze diagonal vs off-diagonal for each fold with subtraction
# ============================================================================
def analyze_cross_modal_by_fold(fold_matrices, fold_runs, fisher=False):
    """
    Compute diagonal and off-diagonal means for each fold, plus their difference.
    
    Returns:
    - DataFrame with columns: fold_run, diagonal_mean, off_diagonal_mean, 
                             scene_specific_similarity (diagonal - off_diagonal)
    """
    results = []
    
    for matrix, run in zip(fold_matrices, fold_runs):
        # Fisher transform if requested
        if fisher:
            matrix = run_fisher(matrix.copy())
        
        # Compute diagonal mean
        diag_mean = np.trace(matrix) / matrix.shape[0]
        
        # Compute off-diagonal mean
        off_diag_mean = np.mean(matrix[~np.eye(len(matrix), dtype=bool)])
        
        # Compute scene-specific similarity (diagonal - off-diagonal)
        scene_specific = diag_mean - off_diag_mean
        
        # Inverse Fisher transform for interpretability
        if fisher:
            diag_mean_orig = inverse_fisher_z(diag_mean)
            off_diag_mean_orig = inverse_fisher_z(off_diag_mean)
            scene_specific_orig = inverse_fisher_z(scene_specific)
            
            results.append({
                'fold_run': run,
                'diagonal_mean': diag_mean_orig,
                'off_diagonal_mean': off_diag_mean_orig,
                'scene_specific_similarity': scene_specific_orig,
                'diagonal_mean_z': diag_mean,
                'off_diagonal_mean_z': off_diag_mean,
                'scene_specific_similarity_z': scene_specific
            })
        else:
            results.append({
                'fold_run': run,
                'diagonal_mean': diag_mean,
                'off_diagonal_mean': off_diag_mean,
                'scene_specific_similarity': scene_specific
            })
    
    return pd.DataFrame(results)
# ============================================================================
# END OF NEW FUNCTIONS
# ============================================================================


# Function to visualise RSA matrix
def visualise_rsa(rsa_corrs, roi):
    # Visualize the RSA matrix using seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(rsa_corrs, center=0, cmap='coolwarm', annot=False, square=True, cbar=True, xticklabels=True,
                yticklabels=True)

    # Add title and labels
    plt.title(f"RSA Matrix Heatmap {roi}", fontsize=16)
    plt.xlabel("Stimulus ID", fontsize=12)
    plt.ylabel("Stimulus ID", fontsize=12)

    plt.tight_layout()
    plt.show()

    return 0


# Function to run entire RSA pipeline
def run_rsa(preproc_dir, sub, roi, nruns, conds, hpc, vis, pos, norm, exclusion_path, apply_exclusion, atlas):
    """
    preproc_dir: Path to the preprocessed directory (not subject specific!)
    sub: The subject to analyse as a string e.g. "04"
    roi: The name of the ROI to use, must be the FILENAME
    nruns: Number of runs to analyse (goes 1 to nruns)
    conds: Conditions to analyse, must be specified as list of strings
    hpc: Boolean that specifies if the ROI is from ASHS (True) or Harvard-Oxford (False - default)
    norm: Boolean specifies whether to perform voxel-wise normalization.
    exclusion_path: path to structure containing trials/blocks to exclusion
    apply_exclusion: Boolean specifying whether to apply exclusion criteria

    Returns the RSA matrix for one subject, one ROI, one interaction.
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
    masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist, hpc, pos, norm, atlas)
    
    if masked_betas is None:
        return None

    # Get the RSA matrix
    rsa_corrs = get_rsa_matrix(conds, condlist, masked_betas)

    # Visualise the matrix
    if vis: visualise_rsa(rsa_corrs, roi)

    return rsa_corrs


# Function to get the right metrics when averaging the entire matrix and not including diagonal for same conditions
def simple_average(rsa_matrix, cond, fisher):
    """
    rsa_matrix: Numpy matrix
    cond: conditions with - in between
    fisher: boolean indicating whether fisher transform is needed
    """
    # Fisher transform the matrix
    if fisher: rsa_matrix = run_fisher(rsa_matrix)

    # Calculate the mean
    if cond.split('-')[0] != cond.split('-')[1]:
        mean = np.mean(rsa_matrix)
        mean_z = mean
        if fisher: mean = inverse_fisher_z(mean)
        t = "with_diag"
    # Exclude the diagonal when the interaction is between the same conditions
    else:
        mean = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
        mean_z = mean
        if fisher: mean = inverse_fisher_z(mean)
        t = "without_diag"

    # Return the averages and types
    if fisher:
        return [mean], [mean_z], [t]
    else:
        return [mean], [t]

# Function to get the right metrics when doing cross-modal analysis (A-V or IC_V-IC_A)
def cross_modal(rsa_matrix, cond, fisher):
    """
    rsa_matrix: Numpy matrix
    fisher: boolean indicating whether fisher transform is needed

    Returns the diagonal mean and off diagonal mean
    """
    # Fisher transform the matrix
    if fisher: rsa_matrix = run_fisher(rsa_matrix)

    # Calculate diagonal mean
    mean_diag = np.trace(rsa_matrix) / rsa_matrix.shape[0]
    mean_dian_z = mean_diag
    if fisher: mean_diag = inverse_fisher_z(mean_diag)

    # Calculate off diagonal mean
    mean_off_diag = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
    mean_off_diag_z = mean_off_diag
    if fisher: mean_off_diag = inverse_fisher_z(mean_off_diag)

    # Return the averages and types
    if fisher: 
        return [mean_diag, mean_off_diag], [mean_dian_z, mean_off_diag_z], ["diag", "off_diag"]
    else:
        return [mean_diag, mean_off_diag], ["diag", "off_diag"]

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


# Loops to run the whole analysis, save the dataframe, and return it
def run_rsa_analysis(rois, rois_bool, pos, subs, nruns, norm, fisher, user, vis, 
                     preproc_dir, conditions, df, analysis, results_path, exclusion_path, apply_exclusion, save_matrix, atlas):
     
    # Initialize dataframe for fold-wise results
    df_foldwise = pd.DataFrame()
    
    # Loop over subjects
    for sub in subs:
        # Loop over conditions
        for cond in conditions:
            t1 = time.time()
            # Loop over ROIs
            for r in range(len(rois)):
                # Calculate the RSA matrix for given subject, condition, and ROI
                rsa_matrix = run_rsa(preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                                     conds=cond.split('-'), hpc=rois_bool[r], vis=vis, pos=pos[r], norm=norm, 
                                     exclusion_path=exclusion_path, apply_exclusion=apply_exclusion, atlas=atlas)
                
                if rsa_matrix is None:
                    continue
                    
                rsa_matrix = rsa_matrix.to_numpy()

                # Calculate and store the metrics + their type
                if fisher:
                    metrics, metrics_z, types = analysis(rsa_matrix, cond, fisher)
                else:
                    metrics, types = analysis(rsa_matrix, cond, fisher)
                    
                if rois[r] == "HPC":
                    roi_name = f'{pos[r]}_{rois[r]}'
                else:
                    roi_name = rois[r]
                
                # Dynamically save the required metrics based on whether we calculate fisher values and if the matrix must be saved
                if fisher and save_matrix:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], metrics_z[m], types[m], rsa_matrix]
                elif fisher and not save_matrix:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], metrics_z[m], types[m]]
                elif not fisher and save_matrix:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], types[m], rsa_matrix]
                else:
                    for m in range(len(metrics)):
                        df.loc[len(df)] = [sub, cond, roi_name, metrics[m], types[m]]
                
                # ============================================================================
                # COMPUTE FOLD-WISE RESULTS FOR ALL CONDITIONS
                # ============================================================================
                # Compute fold-wise for all conditions
                if '-' in cond:
                    # Get the data again for fold-wise analysis
                    if apply_exclusion:
                        exclusion_df = pd.read_csv(exclusion_path)
                        if f'mm{sub}' in exclusion_df['subs'].values:
                            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
                        else:
                            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
                    else:
                        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
                    
                    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub, exclusion_path, apply_exclusion, nruns)
                    masked_betas = extract_betas(sub, preproc_dir, rois[r], betas, condlist, rois_bool[r], pos[r], norm, atlas)
                    
                    if masked_betas is not None:
                        # Get fold-wise RSA matrices
                        fold_matrices, fold_runs = get_rsa_matrix_by_fold(cond.split('-'), condlist, masked_betas)
                        
                        # Analyze each fold
                        fold_results = analyze_cross_modal_by_fold(fold_matrices, fold_runs, fisher=fisher)
                        
                        # Add subject, condition, and ROI info to fold results
                        fold_results['sub'] = sub
                        fold_results['cond'] = cond
                        fold_results['roi'] = roi_name
                        
                        # Append to fold-wise dataframe
                        df_foldwise = pd.concat([df_foldwise, fold_results], ignore_index=True)
                # ============================================================================
            
            # Store the time taken for 1 subject, 1 condition, all ROIs  
            t2 = time.time()

            print(f"Finished all ROIs for {cond} in {sub} in {t2 - t1} seconds", flush=True)

    # Rename some ROIs for simplicity
    df['roi'] = df['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": 'TP',
        "Lateral Occipital Cortex, inferior division": 'LOC'
    })
    
    df_foldwise['roi'] = df_foldwise['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": 'TP',
        "Lateral Occipital Cortex, inferior division": 'LOC'
    })

    # Save both dataframes as CSV
    df.head(10)
    
    # Convert to CSV instead of pickle
    csv_path = results_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    
    # Save fold-wise results as CSV
    if len(df_foldwise) > 0:
        foldwise_csv_path = results_path.replace('.pkl', '_foldwise.csv')
        df_foldwise.to_csv(foldwise_csv_path, index=False)
        print(f"\nFold-wise results saved to: {foldwise_csv_path}", flush=True)
    
    return df, df_foldwise

# Function to visualise final results
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
        # Draw lines between dots of the same subject, ROI, and condition
        for roi in df_vis['roi'].unique():
            sub_df = df_vis[df_vis['roi'] == roi]

            for sub in sub_df['sub'].unique():
                sub_data = sub_df[sub_df['sub'] == sub]

                x_pos = xticks[roi]  # Get correct x position
                x_vals = [x_pos - 0.2, x_pos + 0.2]
                y_vals = sub_data['corr'].values  # Y positions of the dots

                line_color = 'green' if y_vals[0] > y_vals[1] else 'red'

                ax.ax.plot(x_vals, y_vals, color=line_color, linewidth=0.8, alpha=0.7)  # Draw line

    ax.set_axis_labels("ROI", "R")
    ax.despine(left=True)

    # Get the handles and labels from the legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create a legend with labels
    plt.legend(handles, labels)

    # Add a title
    ax.ax.set_title(f"Metrics for {'Sensory' if sensory else 'HPC'} ROIs with {analysis.__name__} analysis")

    return "Plotted successfully"


if __name__ == "__main__":
    ############ Parse CL args ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    p = parser.parse_args()
    
    # Configuration - adjust these parameters as needed
    subs = [p.sub_id]
    
    execute = True

    atlas = "ASHS"

    if atlas == "ASHS":
        rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1", "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1", "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1", "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",  "Superior Temporal Gyrus, posterior division", "Lateral Occipital Cortex, inferior division"]
        
        rois_bool = [0, 0, 0, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        
        pos = ["", "", "", "", "", "", "combined", "combined", "combined", "combined",
           "combined", "combined", "combined", "combined", "left", "right", "", "","",""]
        
    elif atlas == "ASHS_51":
        rois = ["combined_sub_mask_T1", "combined_dgca4_mask_T1", "combined_ca23_mask_T1", "combined_ca1_mask_T1", "combined_HPC_mask_T1", "combined_post_HPC_mask_T1", "combined_ant_HPC_mask_T1"]
        
        rois_bool = [1, 1, 1, 1, 1, 1, 1]
        
        pos = ["", "", "", "", "", "", ""]
        

    elif atlas == "ASHS_maguire":
        rois = ["HPC", "ca_1", "ca_2_3", "dentate_gyrus", "subiculum", "uncus", "Cyst", "pre_parasubiculum", "HPC", "HPC"]
        
        rois_bool = [True, True, True, True, True, True, True, True, True, True]

        pos = ["combined", "combined", "combined", "combined",
               "combined", "combined", "combined", "combined", "left", "right"]
        
    elif atlas == "hippocampus_freesurfer":
        rois = ["Hippocampus_anterior_left", "Hippocampus_anterior_right", "Hippocampus_posterior_left", "Hippocampus_posterior_right"]
        
        rois_bool = [True, True, True, True]

        pos = ["", "", "", ""]

    
    ################ DO NOT EDIT - STATIC ROI ARRAYS FOR DIFFERENT ATLASES - DO NOT EDIT #################
    
    rois_antpost = ["post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1"]
    
    rois_ASHS = ["CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC", "HPC"]
    
    rois_haroxf = ["Temporal Pole", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole", 
                   "Lateral Occipital Cortex, inferior division", "Superior Temporal Gyrus, anterior division", 
                   "Superior Temporal Gyrus, posterior division"]
    
    rois_maguire = ["ca_1", "ca_2_3", "dentate_gyrus", "subiculum", "uncus", "Cyst", "pre_parasubiculum", "HPC", "HPC", "HPC"]
    
    ###################################################################################################################################
    # Make edits below to choose the specific atlases you want
    
    # List of ROI names
    rois = rois_antpost + rois_ASHS + rois_haroxf
    
    # List of ROI atlas indicators (0 = antpost, 1 = ASHS, 2 = HarOxf)
    rois_bool = [0] * len(rois_antpost) + [1] * len(rois_ASHS) + [2] * len(rois_haroxf)
    
    # List of positions for ASHS segmentations
    pos = [""] * len(rois_antpost) + ["combined"] * (len(rois_ASHS) - 2) + ["left", "right"] + [""] * len(rois_haroxf)
    
    ###################################################################################################################################

    # Other constants
    nruns = 9
    norm = False
    fisher = True
    apply_exclusion = True
    save_matrix = True
    user = "or62"
    vis = False
    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed"
    conditions = ['V-V', 'A-A', 'C-C', 'V-IC_V', 'A-IC_A', 'V-A', "A-C", "V-C"]

    # Dataframe to store the final results
    if fisher and save_matrix:
        df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'z_similarity', 'type', 'rsa_matrix'])
    elif fisher and not save_matrix:
        df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'z_similarity', 'type'])
    elif not fisher and save_matrix:
        df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'type', 'rsa_matrix'])
    else:
        df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'type'])
    
    # Specify which metric program you would like to run here
    analysis = cross_modal

    # Path to store the final csv of results
    results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/across_folds/"
    
    results_file_name = f"sub_{subs[0]}_fisher_{fisher}_norm_{norm}_exclusion_{apply_exclusion}_savematrix_{save_matrix}_RSA_Results.pkl"
    
    # Create the directory and any necessary parent directories
    os.makedirs(results_path, exist_ok=True)
    
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'

    # Run the analysis - now returns both dataframes
    if execute: 
        df, df_foldwise = run_rsa_analysis(rois=rois, rois_bool=rois_bool, pos=pos, subs=subs,
                              nruns=nruns, norm=norm, fisher=fisher, user=user,
                              vis=vis, preproc_dir=preproc_dir, conditions=conditions,
                              df=df, analysis=analysis, results_path=results_path+results_file_name,
                              exclusion_path = exclusion_path, apply_exclusion=apply_exclusion, 
                              save_matrix=save_matrix, atlas=atlas)
        
        print(f"\n=== ANALYSIS COMPLETE ===", flush=True)
        print(f"Standard results saved to: {results_path+results_file_name.replace('.pkl', '.csv')}", flush=True)
        print(f"Fold-wise results saved to: {results_path+results_file_name.replace('.pkl', '_foldwise.csv')}", flush=True)