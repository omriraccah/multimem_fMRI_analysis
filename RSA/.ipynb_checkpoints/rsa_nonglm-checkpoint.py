import warnings
import sys
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import nibabel as nib
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from nilearn.image import resample_to_img
from nilearn.image import smooth_img
import nilearn as ni
import time

sns.set(style='white', context='notebook', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")

"""
PROGRAM FLOW:

1. Load the BOLD data
2. Clean the BOLD data
3. Normalise the BOLD data

---- one function does the above ---- (DONE)

4. Load the labels
5. Time shift the labels

---- one function does the above ---- (DONE)

6. Average the data 
7. Reformat data & labels

---- one function does the above ---- (DONE)

8. Mask the BOLD data
9. Conduct RSA

---- many functions do the above ---- (DONE)
"""

# Take in a single volume and its confounds file for a run
# Optionally filter, run linear detrending, mask timeseries
def denoise_img(vol_img, confounds_df, high_pass_filter=None, detrend=True, smooth_fwhm=0):
    # Perform smoothing if enabled
    if smooth_fwhm != 0:
        vol_img = smooth_img(vol_img, smooth_fwhm=smooth_fwhm, mode='nearest')

    # Regress the confounds
    confounds = confounds_df.values
    clean = ni.image.clean_img(imgs=vol_img,
                               detrend=detrend,
                               confounds=confounds,
                               t_r=1.5,
                               standardize=True,
                               high_pass=high_pass_filter)
    return clean


# Function to load the data and regress confounds
def load_denoised_data(preproc_dir, sub, nruns=9):
    denoised_data = []
    for run in range(1, nruns + 1):
        filename = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-{run}_space-T1w_desc-preproc_bold.nii.gz'
        vol_img = ni.image.load_img(filename)

        confound_file = f'{preproc_dir}/sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-{run}_desc-confounds_timeseries.tsv'
        regressors = ['trans_x', 'rot_x', 'trans_y', 'rot_y', 'trans_z', 'rot_z', 'csf', 'global_signal',
                      'white_matter']
        confound_df = pd.read_csv(confound_file, sep='\t')[regressors]

        high_pass_filter = None

        cleaned = denoise_img(vol_img, confound_df, high_pass_filter=high_pass_filter, detrend=True, smooth_fwhm=0)
        denoised_data.append(cleaned.get_fdata())

    denoised_data = np.stack(denoised_data, axis=0)

    return denoised_data


def strip_paths(list_o_paths):
    stripped = []
    for s in list_o_paths:
        if 'Incongruent' not in s:
            stripped.append([int(s.split('/S')[-1].split('_')[0])])
        else:
            temp = [int(s.split('/S')[-1].split('_')[i].replace('S', '')) for i in [0, 1]]
            stripped.append(temp)
    return stripped


def load_labels(preproc_dir, sub, nruns, shift_TR=3):
    # List of list of dataframes
    labels = []

    # Loop over every run we have
    for run in range(1, nruns + 1):

        # Get the file path
        path = os.path.join(preproc_dir, f"sub-mm{sub}/func/sub-mm{sub}_task-multisensorymemory_run-{run}_events.tsv")
        events = pd.read_csv(path, sep='\t')

        # Calculate the durations
        events['duration'] = np.abs(np.round(events['duration'], 1))
        events['tr_duration'] = events['duration'] / 1.5

        # Store the new data format
        data = []

        for i in range(len(events)):

            trial_type = events.iloc[i]['trial_type']
            tr_duration = events.iloc[i]['tr_duration']

            if pd.isna(tr_duration):
                # Extract the duration
                if 'ITI' in trial_type:
                    tr_duration = float(trial_type.split('_')[1].split("'")[0]) / 1.5
                    print(
                        f'We found a nan in sub {sub} run {run} and trial {i}! We fixed {trial_type} to {tr_duration} TRs', flush=True)
                else:
                    tr_duration = 4

            # Calculate the final scan number with the shift
            scan_no = np.round(events.iloc[i]['onset'] / 1.5) + shift_TR

            # Loop through each TR in the event's duration
            for j in range(int(tr_duration)):
                # Append trial type and TR to the data list
                data.append([scan_no + j, trial_type, 1.5, j])

        # Convert to a dataframe
        labs = pd.DataFrame(data, columns=['scan', 'trial_type', 'duration', 'tr'])
        labels.append(labs)

    labels = pd.concat([df.assign(run=i + 1) for i, df in enumerate(labels)], ignore_index=True)

    return labels


# Function to create (non-convolved) design matrices based on events files
def get_shifted_labels(preproc_dir, sub, nruns=9, shift_TR=3):
    # initialize list for storing matrices across runs
    trial_labels = pd.DataFrame()
    trial_labels_cond = pd.DataFrame()
    trial_labels_IC_V = pd.DataFrame()
    trial_labels_IC_A = pd.DataFrame()
    trial_scan_no = pd.DataFrame()

    # initialize list for storing run labels
    run_labels = list()

    ev = load_labels(preproc_dir, sub, nruns, shift_TR)

    for run in range(1, nruns + 1):

        # load events data for current run
        events = ev[ev['run'] == run]

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

        for index, row in events.iterrows():

            if '/Audio/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_A"
                events.loc[index, 'trial_type_cond'] = "A"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"

            elif '/Visual/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_V"
                events.loc[index, 'trial_type_cond'] = "V"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"

            elif '/Congruent/' in events['trial_type_label'][index]:

                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0][0]
                events.loc[index, 'trial_type'] = f"S{num}_C"
                events.loc[index, 'trial_type_cond'] = "C"
                events.loc[index, 'trial_type_IC_V'] = "NOT IC"
                events.loc[index, 'trial_type_IC_A'] = "NOT IC"

            elif '/Incongruent/' in events['trial_type_label'][index]:
                num = strip_paths([events['trial_type_label'][index][2:-2]])
                num = num[0]
                events.loc[index, 'trial_type'] = f"S{num[0]}_S{num[1]}_IC"
                events.loc[index, 'trial_type_IC_V'] = f"S{num[0]}_IC_V"
                events.loc[index, 'trial_type_IC_A'] = f"S{num[1]}_IC_A"
                events.loc[index, 'trial_type_cond'] = "IC"

            run_labels.append(run)

        # save run labels
        trial_labels[run] = events['trial_type']
        trial_labels_IC_V[run] = events['trial_type_IC_V']
        trial_labels_IC_A[run] = events['trial_type_IC_A']
        trial_labels_cond[run] = events['trial_type_cond']
        trial_scan_no[run] = events['scan']

        # add incongruent

    # stack all trials to one column (vertically) for each correspondence to GLM trial betas
    trial_labels = pd.concat([trial_labels, trial_labels.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    trial_scan_no = pd.concat([trial_scan_no, trial_scan_no.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    trial_labels_cond = pd.concat(
        [trial_labels_cond, trial_labels_cond.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)

    trial_labels_IC_V = pd.concat(
        [trial_labels_IC_V, trial_labels_IC_V.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)
    trial_labels_IC_A = pd.concat(
        [trial_labels_IC_A, trial_labels_IC_A.T.stack().reset_index(name='all_runs')['all_runs']], axis=1)

    condlist = pd.DataFrame()
    condlist['scan'] = trial_scan_no['all_runs']
    condlist['conds_spec'] = trial_labels['all_runs']
    condlist['conds_gen'] = trial_labels_cond['all_runs']
    condlist['runs'] = run_labels
    condlist['conds_spec_IC_V'] = trial_labels_IC_V['all_runs']
    condlist['conds_spec_IC_A'] = trial_labels_IC_A['all_runs']

    return condlist


def get_reformatted_data_and_labels(preproc_dir, sub, nruns=9, shift_TR=3):
    data = load_denoised_data(preproc_dir, sub, nruns)  # shift to denoising version later
    labels = get_shifted_labels(preproc_dir, sub, nruns, shift_TR)

    # Initialise final data and labels
    new_data = []
    new_labels = []

    unique_combinations = labels.groupby(['conds_spec', 'runs'])

    # Loop over all unique label & run combinations
    for (conds_spec, run), group in unique_combinations:

        # Get indices for this condition in this run
        matching_idx = labels[(labels['conds_spec'] == conds_spec) & (labels['runs'] == run)]['scan'] - 1
        matching_idx = np.array(matching_idx.to_list(), dtype=int)
        if len(matching_idx) != 4:
            print("Counting error!", flush=True)

        # Average the data
        avg_data = np.mean(data[run - 1][:, :, :, matching_idx], axis=-1)

        # Append to the new data and labels
        new_data.append(avg_data)
        new_labels.append(group.iloc[0][['conds_spec', 'conds_gen', 'runs', 'conds_spec_IC_V', 'conds_spec_IC_A']])

    # Convert data to 4d numpy matrix
    new_data = np.stack(new_data, axis=-1)

    # Convert labels to pd dataframe
    new_labels = pd.DataFrame(new_labels)
    new_labels.reset_index(drop=True, inplace=True)

    # return new data and new labels
    return new_data, new_labels


# Function to apply mask to betas
def extract_betas(sub, preproc_dir, roi, betas, labels, hpc, pos, norm):
    if hpc:
        roi_mask = f"{preproc_dir}/sub-mm{sub}/rois/ASHS/final/func_masks/bin_masks_5B_T1/{pos}_{roi}_mask_T1_non-bin.nii.gz"
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

    if sub == "14" or sub == "19" or sub == "20":
        masked_betas = [row[~np.isnan(row)] for row in masked_betas]
        masked_betas = np.array(masked_betas)

    return masked_betas

# Function to make the RSA matrix
def get_rsa_matrix(conds, condlist, masked_betas, nruns):
    # Now we need to filter, sort, RSA, and visualise
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]

    # Initialise RSA matrix in pandas
    rsa_corrs = pd.DataFrame(index=scene_id_1, columns=scene_id_2)

    # Create a log on comparisons we have done
    log = []

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
                for r in range(1, nruns + 1):
                    for rr in range(1, nruns + 1):
                        # Get run indices
                        s1_test_idx = (s1_labels['runs'] == r).to_list()
                        s2_test_idx = (s2_labels['runs'] == rr).to_list()

                        # Extract the test & train data
                        s1_test_data = s1_data[s1_test_idx]
                        s1_train_data = s1_data[[not x for x in s1_test_idx]]
                        s2_test_data = s2_data[s2_test_idx]
                        s2_train_data = s2_data[[not x for x in s2_test_idx]]

                        # Average the train data
                        s1_train_data = np.sum(s1_train_data, axis=0) / (nruns - 1)
                        s2_train_data = np.sum(s2_train_data, axis=0) / (nruns - 1)

                        # Calculate the correlations
                        corr1 = np.corrcoef(s1_test_data, s2_train_data)[0, 1]
                        corr2 = np.corrcoef(s2_test_data, s1_train_data)[0, 1]

                        # Save the results
                        corrs.append(corr1)
                        corrs.append(corr2)

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
                for run in range(1, nruns + 1):
                    # Extract run mask
                    run_idx = (s_labels['runs'] == run).to_list()

                    # Extract test and train data
                    test_data = s_data[run_idx]
                    train_data = s_data[[not x for x in run_idx]]

                    # Average the train data
                    train_data = np.sum(train_data, axis=0) / (nruns - 1)

                    # Correlate the train and test data
                    corr = np.corrcoef(train_data, test_data)[0, 1]
                    corrs.append(corr)

                # Average the reliability across runs
                mean_corr = np.mean(corrs)

                # Store the reliability measure
                rsa_corrs.loc[s, ss] = mean_corr

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)

    return rsa_corrs


# This one doesn't have the reliability cross validation for the diagonal
def get_rsa_matrix2(conds, condlist, masked_betas, nruns):
    # Now we need to filter, sort, RSA, and visualise
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, 11)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, 11)]

    # Initialise RSA matrix in pandas
    rsa_corrs = pd.DataFrame(index=scene_id_1, columns=scene_id_2)

    # Create a log on comparisons we have done
    log = []

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
                s2_data = masked_betas[ind2]

                # Average the data
                s1_data = np.sum(s1_data, axis=0) / nruns
                s2_data = np.sum(s2_data, axis=0) / nruns

                # Correlate
                corr = np.corrcoef(s1_data, s2_data)[0, 1]

                # Store it
                rsa_corrs.loc[s, ss] = corr

            else:  # Diagonal
                # Store 0
                rsa_corrs.loc[s, ss] = 0

    rsa_corrs = rsa_corrs[rsa_corrs.columns].astype(float)

    return rsa_corrs


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
def run_rsa(dat, lab, preproc_dir, sub, roi, nruns, conds, hpc, vis, pos, norm):
    """
    dat: 4D data numpy array (after all preprocessing)
    lab: pandas dataframe with the labels (after all preprocessing)
    preproc_dir: Path to the preprocessed directory (not subject specific!)
    sub: The subject to analyse as a string e.g. "04"
    roi: The name of the ROI to use, must be the FILENAME
    nruns: Number of runs to analyse (goes 1 to nruns)
    conds: Conditions to analyse, must be specified as list of strings
    hpc: Boolean that specifies if the ROI is from ASHS (True) or Harvard-Oxford (False - default)
    norm: Boolean specifies whether to perform voxel-wise normalization.

    Returns the RSA matrix for one subject, one ROI, one interaction.
    """

    # Mask the data
    masked_dat = extract_betas(sub, preproc_dir, roi, dat, lab, hpc, pos, norm)

    # Get the RSA matrix
    rsa_corrs = get_rsa_matrix(conds, lab, masked_dat, nruns)

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
        if fisher: mean = inverse_fisher_z(mean)
        t = "with_diag"
    # Exclude the diagonal when the interaction is between the same conditions
    else:
        mean = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
        if fisher: mean = inverse_fisher_z(mean)
        t = "without_diag"

    # Return the averages and types
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
    if fisher: mean_diag = inverse_fisher_z(mean_diag)

    # Calculate off diagonal mean
    mean_off_diag = np.mean(rsa_matrix[~np.eye(len(rsa_matrix), dtype=bool)])
    if fisher: mean_off_diag = inverse_fisher_z(mean_off_diag)

    # Return the averages and types
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
def run_rsa_analysis(rois, rois_bool, pos, subs, nruns, norm, fisher, user,
                     vis, preproc_dir, conditions, df, analysis, results_path):
    # Loop over subjects
    for sub in subs:
        # Loop over conditions
        for cond in conditions:
            t1 = time.time()
            # Get the data and labels
            data, labels = get_reformatted_data_and_labels(preproc_dir, sub, nruns)
            # Loop over ROIs
            for r in range(len(rois)):
                # Calculate the RSA matrix for given subject, condition, and ROI
                rsa_matrix = run_rsa(dat=data, lab=labels, preproc_dir=preproc_dir, sub=sub, roi=rois[r], nruns=nruns,
                                     conds=cond.split('-'), hpc=rois_bool[r], vis=vis, pos=pos[r], norm=norm)
                rsa_matrix = rsa_matrix.to_numpy()

                # Calculate and store the metrics + their type
                metrics, types = analysis(rsa_matrix, cond, fisher)
                for m in range(len(metrics)):
                    df.loc[len(df)] = [sub, cond, rois[r], metrics[m], types[m]]

            # Store the time taken for 1 subject, 1 condition, 13 ROIs
            t2 = time.time()

            print(f"Finished all ROIs for {cond} in {sub} in {t2 - t1} seconds", flush=True)

    # Rename some ROIs for simplicity
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
    # Variables for running the full RSA pipeline
    execute = True

    # ROI specifications
    rois = ["Temporal Pole", "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",
            "Superior Temporal Gyrus, anterior division", "Superior Temporal Gyrus, posterior division",
            "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum"]  # Has to be the name the file is saved as

    rois_bool = [False, False, False, False, False, True,
                 True, True, True, True, True, True, True]  # Boolean to indicate if the program should look in ASHS

    pos = ["", "", "", "", "", "combined", "combined", "combined", "combined",
           "combined", "combined", "combined", "combined"]  # Orientation for ASHS outputs: combined/left/right

    # The subject IDs to include
    subs = ["25"]
    # Other constants
    nruns = 9
    norm = False
    fisher = False
    user = "or62"
    vis = False  # Whether to plot rsa matrices
    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed/"
    conditions = ['V-V', 'A-A']  # Always use a - to mark a new condition, has to be a list of interactions.

    # Dataframe to store the final results
    df = pd.DataFrame(columns=['sub', 'cond', 'roi', 'corr', 'type'])

    # Specify which metric program you would like to run here
    analysis = simple_average

    # Path to store the final csv of results
    results_path = f"/gpfs/milgram/scratch60/turk-browne/aa2842/sandbox/rsa_results_subs_{subs[-1]}_fisher_{fisher}_norm_{norm}_func_{analysis.__name__}_testing_nonglm.csv"

    # Run the analysis
    if execute: df = run_rsa_analysis(rois=rois, rois_bool=rois_bool, pos=pos, subs=subs,
                                      nruns=nruns, norm=norm, fisher=fisher, user=user,
                                      vis=vis, preproc_dir=preproc_dir, conditions=conditions,
                                      df=df, analysis=analysis, results_path=results_path)
