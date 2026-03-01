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
    
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()

    betas = results_glmsingle['typed']['betasmd']

    # BEFORE: FULL MODEL
    #results_glmsingle['typed'] = np.load(join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                         #allow_pickle=True).item()

    #betas = results_glmsingle['typed']['betasmd']

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
# loop through runs, create design matrices

# Function to create (non-convolved) design matrices based on events files
def get_labels(preproc_dir, sub, exclusion_path, apply_exclusion, nruns="9"): #OR_edit: changed to load excluded trials for each participant and edited code to exclude blocks and indicate excluded trials. 

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
        #elif atlas == "ASHS_maguire":
        #    roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/{atlas}/final/func_masks/bin_masks_5B_T1/{pos}_{roi}_mask_T1_non-bin.nii.gz" 
        #elif atlas == "hippocampus_freesurfer":
        #   roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/{atlas}/{roi}.nii.gz" # CHANGED TO MGGUIRE
    
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

# Function to make the RSA matrix
# CORRECTED: Fisher transforms each correlation BEFORE averaging
def get_rsa_matrix(conds, condlist, masked_betas):
    
    # Now we need to filter, sort, RSA, and visualise
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]

    # Initialise RSA matrix in pandas
    rsa_corrs = pd.DataFrame(index=scene_id_1, columns=scene_id_2)
    print('PRINT RSA MATRIX INITIALIZED:')
    print(rsa_corrs.to_string())
    
    rsa_fold_counts = pd.DataFrame(index=scene_id_1, columns=scene_id_2)

    # Create a log on comparisons we have done
    log = []
    
    # OR_edit: added to replace all instances of nruns (in case whole runs were deleted)
    runs = condlist['runs'].unique()
    
    # ADDED FOR PRINTING
    #for s in scene_id_1:
    #    for ss in scene_id_2:
    #        print('SCENE LOOPS OUTPUTS:')
    #        print(s)
    #        print(ss)
    
    for s in scene_id_1:
        for ss in scene_id_2:
                
            # Get indices for the scenes we want
            if "IC" in conds[0]:
                ind1 = condlist[f"conds_spec_{conds[0]}"].str.contains(f"{s}").to_list()
            else:
                ind1 = condlist['conds_spec'].str.contains(f'{s}').to_list()

            if "IC" in conds[1]:
                ind2 = condlist[f"conds_spec_{conds[1]}"].str.contains(f"{ss}").to_list()
            else:
                ind2 = condlist['conds_spec'].str.contains(f'{ss}').to_list()
                
            #print(f'SCENE 1:{s}')
            #print(f'SCENE 2:{ss}')

            if s != ss: # Off-diagonal
                # Get the data
                s1_data = masked_betas[ind1]
                s1_labels = condlist[ind1]
                s2_data = masked_betas[ind2]
                s2_labels = condlist[ind2]
                
                #print(f"RUNS FOR SCENE 1: {s1_labels['runs']}")
                #print(f"RUNS FOR SCENE 2: {s2_labels['runs']}")

                # List to store Fisher z-values 
                corrs_z = []
                    
                # Create folds
                for r in runs: # changed to runs in case any excluded
                    
                    # Get run indices
                    s1_test_idx = (s1_labels['runs'] == r).to_list()
                    s2_test_idx = (s2_labels['runs'] == r).to_list()

                    #print(f'TEST INDEX 1:{s1_test_idx}')
                    #print(f'TEST INDEX 2:{s2_test_idx}')

                    # Extract the test & train data
                    s1_test_data = s1_data[s1_test_idx]
                    s1_train_data = s1_data[[not x for x in s1_test_idx]] 
                    s2_test_data = s2_data[s2_test_idx]
                    s2_train_data = s2_data[[not x for x in s2_test_idx]]

                   # print(f'TRAIN INDEX 1:{[not x for x in s1_test_idx]}')
                    #print(f'TRAIN INDEX 2:{[not x for x in s2_test_idx]}')

                    # Average the train data
                    s1_train_data = np.mean(s1_train_data, axis=0) 
                    s2_train_data = np.mean(s2_train_data, axis=0) 

                    corr1 = np.corrcoef(s1_test_data, s2_train_data)
                    corr2 = np.corrcoef(s2_test_data, s1_train_data)

                    # Fisher transform before appending
                    if corr1 != []: # When a trial is excluded the correlation returns an empty matrix.
                        corrs_z.append(fisher_z(corr1[0, 1]))

                    if corr2 != []:
                        corrs_z.append(fisher_z(corr2[0, 1]))

                # Average all Fisher z-values across folds
                # print(f'NUM FOLDS AFTER CORRS: {len(corrs_z)}')
                #print(f'PRINTING Z for off-diag square:{corrs_z}')
                mean_z = np.mean(corrs_z)
                # Store z-value in RSA matrix
                rsa_corrs.loc[s, ss] = mean_z
                rsa_fold_counts.loc[s,ss] = len(corrs_z)

            else:  # Diagonal
                
                #print(f'SCENE 1:{s}')
                #print(f'SCENE 2:{ss}')
                
                # Extract the data
                s_data = masked_betas[ind1]
                s_labels = condlist[ind1]

                # Store the Fisher z-values for each split
                corrs_z = []
                # Perform reliability
                
                #print(f"RUNS FOR SCENE 1: {s_labels['runs']}")
                #print(f"RUNS FOR SCENE 2: {s_labels['runs']}")

                for run in runs:
                    # Extract run mask
                    run_idx = (s_labels['runs'] == run).to_list()

                    # Extract test and train data
                    test_data = s_data[run_idx]
                    train_data = s_data[[not x for x in run_idx]]
                    
                   # print(f'TEST INDEX:{run_idx}')
                   # print(f'TEST DATA SHAPE:{train_data.shape}')
                   # print(f'TRAIN INDEX:{[not x for x in run_idx]}')
                   # print(f'TRAINING DATA SHAPE:{train_data.shape}')

                    # Average the train data
                    train_data = np.mean(train_data, axis=0) # np.sum(train_data, axis=0) / (len(condlist['runs'].unique()) - 1)
                    
                    # Correlate the train and test data
                    corr = np.corrcoef(train_data, test_data) # OR_edit: same changes here to account for left out trials 
                    
                    # Fisher transform BEFORE appending
                    if corr != []:
                        corrs_z.append(fisher_z(corr[0, 1]))

                # Average the Fisher z-values across runs
                mean_z = np.mean(corrs_z)
                # Store z-value (do NOT inverse transform here)
                rsa_corrs.loc[s, ss] = mean_z
                rsa_fold_counts.loc[s,ss] = len(corrs_z)

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)
    print('PRINT RSA MATRIX COMPLETED (z):')
    print(rsa_corrs.to_string())
    print('PRINT RSA MATRIX FOLD COUNTS:')
    print(rsa_fold_counts.to_string())

    return rsa_corrs

def perform_scene_classification(preproc_dir, sub, roi, nruns, cond, hpc, pos, norm, exclusion_path, apply_exclusion):
    """
    Performs scene classification using RSA within a condition and calculates d-prime.
    
    Parameters:
    Same as run_rsa, but cond is a single condition
    
    Returns:
    d_prime: d-prime value calculated from aggregated stats across all scene pairs
    accuracy: classification accuracy from aggregated stats
    individual_results: detailed results for each scene pair
    aggregated_stats: totals for hits, misses, false alarms, correct rejections across all pairs
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
        for j in range(i+1, len(scene_ids)):  # Only unique pairs
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
            
            # For tracking classification performance for this pair
            pair_hits = 0
            pair_misses = 0
            pair_false_alarms = 0
            pair_correct_rejections = 0
            
            # For each run (leave-one-block-out)
            for r in runs:
                # Scene 1 as test case
                test_idx1 = (s1_labels['runs'] == r).to_list()
                
                # Skip if test data is empty
                if sum(test_idx1) == 0:
                    continue
                
                test_data1 = s1_data[test_idx1]
                
                # Check for valid test data
                if len(test_data1) == 0 or np.isnan(test_data1).all():
                    continue
                
                # Get training data for both scenes (excluding test run)
                train_idx1 = (s1_labels['runs'] != r).to_list()
                train_data1 = s1_data[train_idx1]
                
                train_idx2 = (s2_labels['runs'] != r).to_list()
                train_data2 = s2_data[train_idx2]
                
                # Skip if training data is empty
                if len(train_data1) == 0 or np.isnan(train_data1).all() or len(train_data2) == 0 or np.isnan(train_data2).all():
                    continue
                
                # Average training data
                avg_train_data1 = np.mean(train_data1, axis=0)
                avg_train_data2 = np.mean(train_data2, axis=0)
                
                # Calculate correlations
                corr1 = np.corrcoef(test_data1.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data1.flatten(), avg_train_data2.flatten())[0, 1]
                
                # Skip if correlations are invalid
                if np.isnan(corr1) or np.isnan(corr2):
                    continue
                
                # Classify based on maximum correlation
                if corr1 > corr2:  # Correctly identified as scene 1
                    pair_hits += 1
                else:  # Misidentified as scene 2
                    pair_misses += 1
                
                # Scene 2 as test case
                test_idx2 = (s2_labels['runs'] == r).to_list()
                
                # Skip if test data is empty
                if sum(test_idx2) == 0:
                    continue
                
                test_data2 = s2_data[test_idx2]
                
                # Check for valid test data
                if len(test_data2) == 0 or np.isnan(test_data2).all():
                    continue
                
                # Calculate correlations
                corr1 = np.corrcoef(test_data2.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data2.flatten(), avg_train_data2.flatten())[0, 1]
                
                # Skip if correlations are invalid
                if np.isnan(corr1) or np.isnan(corr2):
                    continue
                
                # Classify based on maximum correlation
                if corr1 > corr2:  # Misidentified as scene 1
                    pair_false_alarms += 1
                else:  # Correctly identified as scene 2
                    pair_correct_rejections += 1
            
            # Add pair results to total counts
            total_hits += pair_hits
            total_misses += pair_misses
            total_false_alarms += pair_false_alarms
            total_correct_rejections += pair_correct_rejections
            
            # Calculate metrics for this scene pair
            pair_total = pair_hits + pair_misses + pair_false_alarms + pair_correct_rejections
            
            # Skip storing individual results if no valid classifications for this pair
            if pair_total == 0:
                continue
                
            pair_accuracy = (pair_hits + pair_correct_rejections) / pair_total
            
            # Calculate hit and false alarm rates for this pair
            pair_hit_rate = pair_hits / (pair_hits + pair_misses) if (pair_hits + pair_misses) > 0 else 0.5
            pair_fa_rate = pair_false_alarms / (pair_false_alarms + pair_correct_rejections) if (pair_false_alarms + pair_correct_rejections) > 0 else 0.5
            
            # Adjust rates to avoid infinite d'
            pair_hit_rate = max(min(pair_hit_rate, 0.999), 0.001)
            pair_fa_rate = max(min(pair_fa_rate, 0.999), 0.001)
            
            # Calculate d-prime for this pair
            pair_d_prime = stats.norm.ppf(pair_hit_rate) - stats.norm.ppf(pair_fa_rate)
            
            # Store individual pair results
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
    
    # Calculate metrics from aggregated statistics
    total_trials = total_hits + total_misses + total_false_alarms + total_correct_rejections
    
    if total_trials > 0:
        # Calculate overall accuracy
        aggregated_accuracy = (total_hits + total_correct_rejections) / total_trials
        
        # Calculate hit and false alarm rates from aggregated statistics
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.5
        fa_rate = total_false_alarms / (total_false_alarms + total_correct_rejections) if (total_false_alarms + total_correct_rejections) > 0 else 0.5
        
        # Adjust rates to avoid infinite d'
        hit_rate = max(min(hit_rate, 0.999), 0.001)
        fa_rate = max(min(fa_rate, 0.999), 0.001)
        
        # Calculate d-prime from aggregated statistics
        aggregated_d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    else:
        aggregated_accuracy = np.nan
        aggregated_d_prime = np.nan
    
    # Create dictionary with aggregated statistics
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

# [Other functions continue - perform_scene_classification_scene_level, run_scene_classification_analysis, etc.]
# [Remaining code is identical to original...]


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
            
            print("PRINTING SCENE IDs:")
            print(f'Scene 1:{scene1}')
            print(f'Scene 2:{scene2}')
            
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
            
            print("PRINTING SCENE 1 labels:")
            print(f'Scene 1:{scene1}')
            
            s2_data = masked_betas[s2_indices]
            s2_labels = condlist[s2_indices]
            
            hits = 0
            misses = 0
            false_alarms = 0
            correct_rejections = 0
            
            for r in runs:
                test_idx1 = (s1_labels['runs'] == r).to_list()
                
                if sum(test_idx1) == 0:
                    print("TEST INDEX EMPTY -- this should never be the case")
                    continue
                
                test_data1 = s1_data[test_idx1]
                
                if len(test_data1) == 0 or np.isnan(test_data1).all():
                    print("TEST DATA EMPTY -- this should never be the case")
                    continue
                
                train_idx1 = (s1_labels['runs'] != r).to_list()
                train_data1 = s1_data[train_idx1]
                
                train_idx2 = (s2_labels['runs'] != r).to_list()
                train_data2 = s2_data[train_idx2]
                
                if len(train_data1) == 0 or np.isnan(train_data1).all() or len(train_data2) == 0 or np.isnan(train_data2).all():
                    print("TRAIN DATA EMPTY -- this should never be the case")
                    continue
                
                avg_train_data1 = np.mean(train_data1, axis=0)
                avg_train_data2 = np.mean(train_data2, axis=0)
                
                corr1 = np.corrcoef(test_data1.flatten(), avg_train_data1.flatten())[0, 1]
                corr2 = np.corrcoef(test_data1.flatten(), avg_train_data2.flatten())[0, 1]
                
                if np.isnan(corr1) or np.isnan(corr2):
                    print("CORRELATION HAS NAN -- this should never be the case")
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
            print("PRINTING TOTAL")
            
            if total == 0:
                print("NO VALID CLASSIFICATION")
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

def run_rsa(preproc_dir, sub, roi, nruns, conds, hpc, vis, pos, norm, exclusion_path, apply_exclusion, atlas):
    """Run entire RSA pipeline"""
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
    rsa_corrs = get_rsa_matrix(conds, condlist, masked_betas)

    if vis: visualise_rsa(rsa_corrs, roi)

    return rsa_corrs

def simple_average(rsa_matrix, cond, fisher):
    """Get the right metrics when averaging the entire matrix"""
    # NOTE: rsa_matrix already contains Fisher z-values from get_rsa_matrix()
    # Do NOT Fisher transform again!
    
    if cond.split('-')[0] != cond.split('-')[1]:
        mean_z = np.mean(rsa_matrix)
        t = "with_diag"
    else:
        mean_z = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
        t = "without_diag"
    
    # Inverse transform to r-value for reporting
    if fisher:
        mean = inverse_fisher_z(mean_z)
        return [mean], [mean_z], [t]
    else:
        return [inverse_fisher_z(mean_z)], [t]

def cross_modal(rsa_matrix, cond, fisher):
    """Get the right metrics when doing cross-modal analysis"""
    # NOTE: rsa_matrix already contains Fisher z-values from get_rsa_matrix()
    # Do NOT Fisher transform again!
    
    # Calculate diagonal mean (already in z-space)
    mean_diag_z = np.trace(rsa_matrix) / rsa_matrix.shape[0]
    print(f'DIAG MEAN Z: {mean_diag_z}')

    # Calculate off diagonal mean (already in z-space)
    mean_off_diag_z = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
    print(f'OFF-DIAG MEAN Z: {mean_off_diag_z}')
    
    # Inverse transform to r-values for reporting
    if fisher:
        mean_diag = inverse_fisher_z(mean_diag_z)
        mean_off_diag = inverse_fisher_z(mean_off_diag_z)
        return [mean_diag, mean_off_diag], [mean_diag_z, mean_off_diag_z], ["diag", "off_diag"]
    else:
        # If fisher=False, the matrix should still contain z-values
        # but we return both z and r
        return [inverse_fisher_z(mean_diag_z), inverse_fisher_z(mean_off_diag_z)], ["diag", "off_diag"]

def run_rsa_analysis(rois, rois_bool, pos, subs, nruns, norm, fisher, user,
                     vis, preproc_dir, conditions, df, analysis, results_path, exclusion_path, apply_exclusion, atlas):
     
    for sub in subs:
        for cond in conditions:
            t1 = time.time()
            for r in range(len(rois)):
                rsa_matrix = run_rsa(preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                                     conds=cond.split('-'), hpc=rois_bool[r], vis=vis, pos=pos[r], norm=norm, exclusion_path=exclusion_path, apply_exclusion=apply_exclusion, atlas=atlas)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--sub_id',type=str)
    p = parser.parse_args()
    
    subs = [p.sub_id]
    execute = True
    atlas = "ASHS"

    if atlas == "ASHS":
        #rois= ["post_right_HPC_mask_T1"]
        #rois_bool = [0]
        #pos = [""]
        rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1", "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1", "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1", "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum", "HPC", "HPC", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",  "Superior Temporal Gyrus, posterior division", "Lateral Occipital Cortex, inferior division"]
        rois_bool = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        pos = ["", "", "", "", "", "", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "left", "right", "", "","",""]
        
    #elif atlas == "ASHS_51":
        #rois = ["combined_sub_mask_T1", "combined_dgca4_mask_T1", "combined_ca23_mask_T1", "combined_ca1_mask_T1", #"combined_HPC_mask_T1", "combined_post_HPC_mask_T1", "combined_ant_HPC_mask_T1"]
    #    rois_bool = [1, 1, 1, 1, 1, 1, 1]
    #    pos = ["", "", "", "", "", "", ""]
        
    #elif atlas == "ASHS_maguire":
     #   rois = ["HPC", "ca_1", "ca_2_3", "dentate_gyrus", "subiculum", "uncus", "Cyst", "pre_parasubiculum", "HPC", "HPC"]
      #  rois_bool = [True, True, True, True, True, True, True, True, True, True]
     #   pos = ["combined", "combined", "combined", "combined", "combined", "combined", "combined", "combined", "left", "right"]
        
    #elif atlas == "hippocampus_freesurfer":
    #    rois = ["Hippocampus_anterior_left", "Hippocampus_anterior_right", "Hippocampus_posterior_left", "Hippocampus_posterior_right"]
   #     rois_bool = [True, True, True, True]
  #      pos = ["", "", "", ""]

    nruns = 9
    norm = False
    fisher = True
    apply_exclusion = True
    user = "or62"
    vis = False
    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed"
    conditions = ['V-V', 'A-A', 'C-C', 'V-IC_V', 'A-IC_A', 'V-A', 'IC_V-IC_V', 'IC_A-IC_A']
    conditions_classification = ['V', 'A', 'C', 'IC_V', 'IC_A']

    df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'z_similarity', 'type'])
    df_classification = pd.DataFrame(columns=['sub', 'cond', 'roi', 'd_prime', 'accuracy', 'individual results','num_scene_pairs', 'aggregated_stats'])

    analysis = cross_modal

    results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/{atlas}/"
    results_file_name = f"sub_{subs[0]}_atlas_{atlas}_fisher_{fisher}_norm_{norm}_exclusion_{apply_exclusion}_RSA_Results_singleloop_glmC.csv"
    
    os.makedirs(results_path, exist_ok=True)
    
    classification_results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/scene_classification_subs_{len(subs)}_fisher_{fisher}_norm_{norm}_exclusion_{apply_exclusion}_Results_singleloop_glmC.csv"
   
    exclusion_path = '/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv'

    if execute: df = run_rsa_analysis(rois=rois, rois_bool=rois_bool, pos=pos, subs=subs,
                                      nruns=nruns, norm=norm, fisher=fisher, user=user,
                                      vis=vis, preproc_dir=preproc_dir, conditions=conditions,
                                      df=df, analysis=analysis, results_path=results_path+results_file_name, exclusion_path = exclusion_path, apply_exclusion=apply_exclusion, atlas=atlas)