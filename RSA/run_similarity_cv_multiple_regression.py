#!/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem/bin/python
"""
Similarity-matrix CV multiple regression: testing multimodal representation
in the human hippocampus beyond unimodal coding.

The similarity matrices here are Fisher-z correlations (similarities, not
dissimilarities), so this is technically RSM regression rather than RDM
regression. The model asks whether the multimodal (C-C) similarity
structure can be explained by the unimodal (A-A, V-V) similarity structure
plus a movie-specific multimodal-coding component (identity regressor).

For each subject and ROI, the regression is run two ways:

  (1) Cross-validated (leave-one-fold-out, primary):
      a. Build per-fold similarity matrices for A-A, V-V, C-C
         (shape: n_folds x 10 x 10), mirroring the cross-validation in
         get_rsa_matrix without averaging across folds.
      b. For each held-out fold k, fit OLS on the other 8 folds:
            vec(C-C) = b0 + b_A*vec(A-A) + b_V*vec(V-V) + b_MM*vec(I) + eps
         Apply b to fold k. Compute residuals.
      c. Aggregate residuals across folds (nanmean per cell) and report
         CV betas, CV R^2, and residual MSPS.

  (2) Averaged-matrix (no CV, comparison):
      Collapse the per-fold matrices to one 10x10 per condition (nanmean,
      so excluded folds drop out cell-by-cell). Fit a single OLS on the
      100 cells. Reports in-sample betas, R^2, and residual MSPS.

Trial exclusions are handled naturally via the existing get_labels
relabeling: excluded trials become "exclude_N" and silently fail to match
any condition pattern. Cells that cannot be computed for a given fold
remain NaN and are dropped from the regression for that fold via a
finiteness mask.

CLI usage:
    run_similarity_cv_multiple_regression.py -sub 01
"""

import warnings
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------------------------------------------
# Import shared functions from main RSA script
# The RSA module lives in the home-dir copy of the project, while this
# script lives in /gpfs/milgram/pi/turk-browne/or62/multisensory-memory-project/
# Similarity_regression/, so we add the RSA module's parent dir explicitly
# rather than using a relative '..' import.
# ------------------------------------------------------------------
RSA_PROJECT_ROOT = '/gpfs/milgram/home/or62/project/multisensory-memory-project'
sys.path.insert(0, RSA_PROJECT_ROOT)
from RSA.run_rsa_w_exclusion import (
    load_glm_single,
    get_labels,
    extract_betas,
    fisher_z,
)


# ------------------------------------------------------------------
# Per-fold cross-validated similarity matrix
# ------------------------------------------------------------------
def get_per_fold_matrix(conds, condlist, masked_betas, n_scenes=10):
    """
    Per-fold cross-validated similarity matrix for one condition pair.

    Mirrors the leave-one-run-out logic in get_rsa_matrix
    (run_rsa_w_exclusion.py) but keeps each fold separate. Returns a
    (n_folds, n_scenes, n_scenes) array of Fisher-z correlations, with
    NaN cells where a fold could not be computed (due to trial exclusions
    on both train and test sides for that cell).

    For off-diagonal cells, each fold value averages the two
    cross-correlations within that fold (s1_test vs s2_train and
    s2_test vs s1_train), matching the symmetry used in get_rsa_matrix.

    Parameters
    ----------
    conds : list of two strings, e.g. ['A', 'A'], ['V', 'V'], ['C', 'C'],
        or ['V', 'A'] for the cross-modal matrix.
    condlist : DataFrame
        Output of get_labels (one row per trial, columns conds_spec/runs/etc).
    masked_betas : ndarray, shape (n_trials, n_voxels)
        ROI-masked single-trial beta patterns from GLMSingle.
    n_scenes : int, default 10

    Returns
    -------
    per_fold : ndarray, shape (n_folds, n_scenes, n_scenes), dtype float64.
        Fisher-z correlations. NaN where the cell could not be computed.
    """
    scene_id_1 = [f"S{i}_{conds[0]}" for i in range(1, n_scenes + 1)]
    scene_id_2 = [f"S{i}_{conds[1]}" for i in range(1, n_scenes + 1)]

    runs = list(condlist['runs'].unique())
    n_folds = len(runs)
    out = np.full((n_folds, n_scenes, n_scenes), np.nan)

    for fold_idx, r in enumerate(runs):
        for i, s in enumerate(scene_id_1):
            for j, ss in enumerate(scene_id_2):

                # Index trials matching scene/condition
                if "IC" in conds[0]:
                    ind1 = condlist[f"conds_spec_{conds[0]}"].str.contains(f"{s}").to_list()
                else:
                    ind1 = condlist['conds_spec'].str.contains(f'{s}').to_list()

                if "IC" in conds[1]:
                    ind2 = condlist[f"conds_spec_{conds[1]}"].str.contains(f"{ss}").to_list()
                else:
                    ind2 = condlist['conds_spec'].str.contains(f'{ss}').to_list()

                s1_data = masked_betas[ind1]
                s1_labels = condlist[ind1]
                s2_data = masked_betas[ind2]
                s2_labels = condlist[ind2]

                if s != ss:
                    # ---- off-diagonal cell ----
                    s1_test_idx = (s1_labels['runs'] == r).to_list()
                    s2_test_idx = (s2_labels['runs'] == r).to_list()

                    s1_test_data = s1_data[s1_test_idx]
                    s1_train_data = s1_data[[not x for x in s1_test_idx]]
                    s2_test_data = s2_data[s2_test_idx]
                    s2_train_data = s2_data[[not x for x in s2_test_idx]]

                    fold_z = []

                    if len(s1_test_data) > 0 and len(s2_train_data) > 0:
                        s2_train_mean = np.mean(s2_train_data, axis=0)
                        c = np.corrcoef(s1_test_data, s2_train_mean)
                        if c.size > 0 and not np.isnan(c[0, 1]):
                            fold_z.append(fisher_z(c[0, 1]))

                    if len(s2_test_data) > 0 and len(s1_train_data) > 0:
                        s1_train_mean = np.mean(s1_train_data, axis=0)
                        c = np.corrcoef(s2_test_data, s1_train_mean)
                        if c.size > 0 and not np.isnan(c[0, 1]):
                            fold_z.append(fisher_z(c[0, 1]))

                    if len(fold_z) > 0:
                        out[fold_idx, i, j] = float(np.mean(fold_z))

                else:
                    # ---- diagonal cell ----
                    s_data = masked_betas[ind1]
                    s_labels = condlist[ind1]
                    run_idx = (s_labels['runs'] == r).to_list()
                    test_data = s_data[run_idx]
                    train_data = s_data[[not x for x in run_idx]]

                    if len(test_data) > 0 and len(train_data) > 0:
                        train_mean = np.mean(train_data, axis=0)
                        c = np.corrcoef(train_mean, test_data)
                        if c.size > 0 and not np.isnan(c[0, 1]):
                            out[fold_idx, i, j] = float(fisher_z(c[0, 1]))

    return out


# ------------------------------------------------------------------
# Similarity-matrix CV multiple regression
# ------------------------------------------------------------------
def similarity_cv_multiple_regression(AA, VV, CC, n_scenes=10, min_train_cells=10):
    """
    Multiple regression of C-C similarity on A-A, V-V. We compute these in two ways for a sanity check (only [1] the cross-validated method is used for subsequent analysis.

    (1) Cross-validated (leave-one-fold-out, primary):
        For each held-out fold k:
            - Stack the other folds' vectorized matrices, drop NaN cells.
            - Fit OLS: vec(C) = b0 + b_A*vec(A) + b_V*vec(V) + b_MM*vec(I) + eps
            - Apply b to fold k, compute residual matrix (NaN where any of
              A_k, V_k, C_k is NaN at that cell).
        Aggregate residuals across folds with nanmean; compute residual MSPS
        and a held-out R^2.

    (2) Averaged-matrix (no CV, comparison):
        Average the per-fold matrices to a single 10x10 per condition
        (nanmean, so excluded cells are dropped). Fit one OLS on the 100
        cells. Compute residual MSPS on those 100 cells.

    Parameters
    ----------
    AA, VV, CC : ndarray, shape (n_folds, n_scenes, n_scenes)
        Per-fold Fisher-z similarity matrices from get_per_fold_matrix.
    n_scenes : int, default 10
    min_train_cells : int, default 10
        Minimum finite training cells required for a CV fold. Folds with
        fewer training cells are skipped.

    Two regression models are fit at each step:
      * full model:    C = b0 + b_A·A + b_V·V + b_MM·I
      * unimodal-only: C = b0 + b_A·A + b_V·V
    The full model gives b_MM (multimodal-specific coefficient). The unimodal-
    only model gives the R^2 that tells the reader how much of C is explained
    by A and V alone, AND its residual matrix is the one used for residual
    MSPS — i.e., "after regressing out the unimodal channels, is the
    diagonal still bigger than the off-diagonal?"  (NB: residuals from the
    full model are guaranteed to have diag − off ≈ 0 by OLS orthogonality
    against I, so that quantity isn't reported.)

    Returns
    -------
    dict with keys:
        # ---- CV outputs ----
        'betas_per_fold'      : (n_folds, 4) — full-model betas per fold
        'betas_per_fold_noI'  : (n_folds, 3) — unimodal-only betas per fold
        'beta_cv'             : (4,)         — mean full-model betas across folds
        'beta_cv_noI'         : (3,)         — mean unimodal-only betas across folds
        'resid_matrix_cv'     : (n_scenes, n_scenes) — full-model residual matrix (visualization)
        'resid_matrix_noI_cv' : (n_scenes, n_scenes) — unimodal-only residual matrix (the meaningful one)
        'resid_msps_noI_cv'   : float — diag − off of resid_matrix_noI_cv
        'cv_r2'               : float — held-out R^2 of full model
        'cv_r2_unimodal'      : float — held-out R^2 of unimodal-only model
        'n_folds_used'        : int
        'cell_fold_counts'    : (n_scenes, n_scenes)
        # ---- Averaged-matrix outputs (no CV) ----
        'beta_avg'            : (4,) — full-model OLS betas on averaged matrices
        'beta_avg_noI'        : (3,) — unimodal-only OLS betas on averaged matrices
        'resid_matrix_noI_avg': (n_scenes, n_scenes) — unimodal-only residual matrix
        'resid_msps_noI_avg'  : float — diag − off
        'avg_r2'              : float — in-sample R^2 of full model
        'avg_r2_unimodal'     : float — in-sample R^2 of unimodal-only model
    """
    n_folds = AA.shape[0]
    assert AA.shape == VV.shape == CC.shape, (
        f"Shape mismatch: AA={AA.shape}, VV={VV.shape}, CC={CC.shape}"
    )

    I_mat = np.eye(n_scenes)
    diag_mask = np.eye(n_scenes, dtype=bool)

    # CV storage
    betas_with_I = np.full((n_folds, 4), np.nan)  # [b0, b_A, b_V, b_MM]
    betas_no_I   = np.full((n_folds, 3), np.nan)  # [b0', b_A', b_V']  (unimodal-only model)
    resids_with_I = np.full((n_folds, n_scenes, n_scenes), np.nan)
    resids_no_I   = np.full((n_folds, n_scenes, n_scenes), np.nan)
    obs_held       = []
    pred_full_held = []
    pred_noI_held  = []
    n_folds_used = 0

    for k in range(n_folds):
        # Build training design from folds != k. Build both with-I and no-I
        # versions in the same pass.
        X_full_list = []
        X_noI_list  = []
        y_list      = []
        for kk in range(n_folds):
            if kk == k:
                continue
            aa = AA[kk].ravel()
            vv = VV[kk].ravel()
            cc = CC[kk].ravel()
            mm = I_mat.ravel()
            ok = np.isfinite(aa) & np.isfinite(vv) & np.isfinite(cc)
            if ok.sum() == 0:
                continue
            ones = np.ones(int(ok.sum()))
            X_full_list.append(np.column_stack([ones, aa[ok], vv[ok], mm[ok]]))
            X_noI_list.append( np.column_stack([ones, aa[ok], vv[ok]]))
            y_list.append(cc[ok])

        if len(y_list) == 0:
            continue
        X_full_train = np.vstack(X_full_list)
        X_noI_train  = np.vstack(X_noI_list)
        y_train      = np.concatenate(y_list)
        if X_full_train.shape[0] < min_train_cells:
            continue

        beta_full, *_ = np.linalg.lstsq(X_full_train, y_train, rcond=None)
        beta_noI,  *_ = np.linalg.lstsq(X_noI_train,  y_train, rcond=None)
        betas_with_I[k] = beta_full
        betas_no_I[k]   = beta_noI

        # Apply to fold k
        aa_k = AA[k]; vv_k = VV[k]; cc_k = CC[k]
        ok_k = np.isfinite(aa_k) & np.isfinite(vv_k) & np.isfinite(cc_k)
        if ok_k.sum() == 0:
            continue
        pred_full_k = beta_full[0] + beta_full[1] * aa_k + beta_full[2] * vv_k + beta_full[3] * I_mat
        pred_noI_k  = beta_noI[0]  + beta_noI[1]  * aa_k + beta_noI[2]  * vv_k

        rmat_full = np.full_like(cc_k, np.nan)
        rmat_noI  = np.full_like(cc_k, np.nan)
        rmat_full[ok_k] = cc_k[ok_k] - pred_full_k[ok_k]
        rmat_noI[ok_k]  = cc_k[ok_k] - pred_noI_k[ok_k]
        resids_with_I[k] = rmat_full
        resids_no_I[k]   = rmat_noI

        obs_held.append(cc_k[ok_k])
        pred_full_held.append(pred_full_k[ok_k])
        pred_noI_held.append(pred_noI_k[ok_k])
        n_folds_used += 1

    # Aggregate residual matrices across folds
    resid_matrix_cv     = np.nanmean(resids_with_I, axis=0)  # for figures only
    resid_matrix_noI_cv = np.nanmean(resids_no_I,   axis=0)  # the meaningful one
    cell_fold_counts    = np.sum(np.isfinite(resids_with_I), axis=0)

    # Residual MSPS from no-I CV residuals
    # (The with-I version is trivially ≈0 because residuals from an OLS that
    #  includes vec(I) are orthogonal to vec(I) by construction.)
    diag_r = np.diag(resid_matrix_noI_cv)
    off_r  = resid_matrix_noI_cv[~diag_mask]
    resid_msps_noI_cv = float(np.nanmean(diag_r) - np.nanmean(off_r))

    # Cross-validated R^2 for the full model and the unimodal-only model
    if obs_held:
        obs       = np.concatenate(obs_held)
        pred_full = np.concatenate(pred_full_held)
        pred_noI  = np.concatenate(pred_noI_held)
        ss_tot    = float(np.sum((obs - np.mean(obs)) ** 2))
        ss_res_full = float(np.sum((obs - pred_full) ** 2))
        ss_res_noI  = float(np.sum((obs - pred_noI)  ** 2))
        cv_r2_full     = 1.0 - ss_res_full / ss_tot if ss_tot > 0 else np.nan
        cv_r2_unimodal = 1.0 - ss_res_noI  / ss_tot if ss_tot > 0 else np.nan
    else:
        cv_r2_full = np.nan
        cv_r2_unimodal = np.nan

    # ------------------------------------------------------------------
    # Averaged-matrix (no CV) regression — collapse to 10x10 first
    # ------------------------------------------------------------------
    AA_mean = np.nanmean(AA, axis=0)
    VV_mean = np.nanmean(VV, axis=0)
    CC_mean = np.nanmean(CC, axis=0)

    aa_v = AA_mean.ravel(); vv_v = VV_mean.ravel(); cc_v = CC_mean.ravel()
    mm_v = I_mat.ravel()
    ok_avg = np.isfinite(aa_v) & np.isfinite(vv_v) & np.isfinite(cc_v)

    if ok_avg.sum() >= 4:
        ones = np.ones(int(ok_avg.sum()))
        X_full = np.column_stack([ones, aa_v[ok_avg], vv_v[ok_avg], mm_v[ok_avg]])
        X_noI  = np.column_stack([ones, aa_v[ok_avg], vv_v[ok_avg]])
        y_avg = cc_v[ok_avg]

        beta_avg_full, *_ = np.linalg.lstsq(X_full, y_avg, rcond=None)
        beta_avg_noI,  *_ = np.linalg.lstsq(X_noI,  y_avg, rcond=None)
        pred_full_avg = X_full @ beta_avg_full
        pred_noI_avg  = X_noI  @ beta_avg_noI

        ss_tot_avg     = float(np.sum((y_avg - np.mean(y_avg)) ** 2))
        ss_res_full    = float(np.sum((y_avg - pred_full_avg) ** 2))
        ss_res_noI     = float(np.sum((y_avg - pred_noI_avg)  ** 2))
        avg_r2_full     = 1.0 - ss_res_full / ss_tot_avg if ss_tot_avg > 0 else np.nan
        avg_r2_unimodal = 1.0 - ss_res_noI  / ss_tot_avg if ss_tot_avg > 0 else np.nan

        # Residual matrix from the NO-I averaged regression (the meaningful one)
        resid_v = np.full(n_scenes * n_scenes, np.nan)
        resid_v[ok_avg] = y_avg - pred_noI_avg
        resid_matrix_noI_avg = resid_v.reshape(n_scenes, n_scenes)
        diag_v = np.diag(resid_matrix_noI_avg)
        off_v  = resid_matrix_noI_avg[~diag_mask]
        resid_msps_noI_avg = float(np.nanmean(diag_v) - np.nanmean(off_v))
    else:
        beta_avg_full = np.full(4, np.nan)
        beta_avg_noI  = np.full(3, np.nan)
        avg_r2_full = np.nan
        avg_r2_unimodal = np.nan
        resid_matrix_noI_avg = np.full((n_scenes, n_scenes), np.nan)
        resid_msps_noI_avg = np.nan

    return {
        # CV outputs (with-I model — primary)
        'betas_per_fold':      betas_with_I,
        'betas_per_fold_noI':  betas_no_I,
        'beta_cv':             np.nanmean(betas_with_I, axis=0),  # [b0, bA, bV, bMM]
        'beta_cv_noI':         np.nanmean(betas_no_I,   axis=0),  # [b0', bA', bV']
        'resid_matrix_cv':     resid_matrix_cv,                   # full-model residuals (~0 by construction in expectation; for viz only)
        'resid_matrix_noI_cv': resid_matrix_noI_cv,               # meaningful residual matrix
        'resid_msps_noI_cv':   resid_msps_noI_cv,                 # diag−off of no-I CV residuals
        'cv_r2':               float(cv_r2_full)     if np.isfinite(cv_r2_full)     else np.nan,
        'cv_r2_unimodal':      float(cv_r2_unimodal) if np.isfinite(cv_r2_unimodal) else np.nan,
        'n_folds_used':        int(n_folds_used),
        'cell_fold_counts':    cell_fold_counts,
        # Averaged-matrix outputs (no CV)
        'beta_avg':            beta_avg_full,                     # full-model: [b0, bA, bV, bMM]
        'beta_avg_noI':        beta_avg_noI,                      # unimodal-only: [b0', bA', bV']
        'resid_matrix_noI_avg':resid_matrix_noI_avg,              # meaningful residual matrix
        'resid_msps_noI_avg':  float(resid_msps_noI_avg) if np.isfinite(resid_msps_noI_avg) else np.nan,
        'avg_r2':              float(avg_r2_full)      if np.isfinite(avg_r2_full)      else np.nan,
        'avg_r2_unimodal':     float(avg_r2_unimodal)  if np.isfinite(avg_r2_unimodal)  else np.nan,
    }


# ------------------------------------------------------------------
# Top-level analysis loop (subject x ROI)
# ------------------------------------------------------------------
def run_cv_regression_analysis(rois, rois_bool, pos, subs, nruns, norm,
                               preproc_dir, exclusion_path, apply_exclusion,
                               atlas, results_csv_path, matrices_dir):
    """Iterate over subjects x ROIs, run similarity CV multiple regression, write outputs."""
    rows = []
    os.makedirs(matrices_dir, exist_ok=True)

    for sub in subs:
        # Load GLMSingle betas (one set per subject)
        if apply_exclusion:
            exclusion_df = pd.read_csv(exclusion_path)
            if f'mm{sub}' in exclusion_df['subs'].values:
                betas = load_glm_single(
                    f"{preproc_dir}/glm_single_results_mm{sub}_excluded/"
                )
            else:
                betas = load_glm_single(
                    f"{preproc_dir}/glm_single_results_mm{sub}/"
                )
        else:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

        condlist = get_labels(
            f"{preproc_dir}/sub-mm{sub}/func/", sub,
            exclusion_path, apply_exclusion, nruns,
        )

        for r in range(len(rois)):
            t1 = time.time()
            try:
                masked_betas = extract_betas(
                    sub, preproc_dir, rois[r], betas,
                    condlist, rois_bool[r], pos[r], norm, atlas,
                )
            except Exception as e:
                print(
                    f"[skip] sub-{sub} ROI {rois[r]}: extract_betas failed ({e})",
                    flush=True,
                )
                continue

            # Per-fold matrices for the three core conditions
            AA = get_per_fold_matrix(['A', 'A'], condlist, masked_betas)
            VV = get_per_fold_matrix(['V', 'V'], condlist, masked_betas)
            CC = get_per_fold_matrix(['C', 'C'], condlist, masked_betas)

            out = similarity_cv_multiple_regression(AA, VV, CC)

            roi_name = f'{pos[r]}_{rois[r]}' if rois[r] == "HPC" else rois[r]

            rows.append({
                'sub':                  sub,
                'roi':                  roi_name,
                # CV (leave-one-fold-out) — full model with identity regressor
                'beta_0_cv':            out['beta_cv'][0],
                'beta_A_cv':            out['beta_cv'][1],
                'beta_V_cv':            out['beta_cv'][2],
                'beta_MM_cv':           out['beta_cv'][3],
                'cv_r2_full':           out['cv_r2'],          # variance of C explained by A+V+I
                'cv_r2_unimodal':       out['cv_r2_unimodal'], # variance of C explained by A+V alone (reviewer's number)
                # CV no-I residual MSPS (reviewer's "still > 0 after regressing out unimodal")
                'beta_0_cv_noI':        out['beta_cv_noI'][0],
                'beta_A_cv_noI':        out['beta_cv_noI'][1],
                'beta_V_cv_noI':        out['beta_cv_noI'][2],
                'resid_msps_noI_cv':    out['resid_msps_noI_cv'],
                # Averaged-matrix (no CV) — full model
                'beta_0_avg':           out['beta_avg'][0],
                'beta_A_avg':           out['beta_avg'][1],
                'beta_V_avg':           out['beta_avg'][2],
                'beta_MM_avg':          out['beta_avg'][3],
                'avg_r2_full':          out['avg_r2'],
                'avg_r2_unimodal':      out['avg_r2_unimodal'],
                # Averaged-matrix no-I residual MSPS
                'beta_0_avg_noI':       out['beta_avg_noI'][0],
                'beta_A_avg_noI':       out['beta_avg_noI'][1],
                'beta_V_avg_noI':       out['beta_avg_noI'][2],
                'resid_msps_noI_avg':   out['resid_msps_noI_avg'],
                # QA
                'n_folds':              AA.shape[0],
                'n_folds_used':         out['n_folds_used'],
                'mean_cell_fold_count': float(np.mean(out['cell_fold_counts'])),
                'min_cell_fold_count':  int(np.min(out['cell_fold_counts'])),
                'max_cell_fold_count':  int(np.max(out['cell_fold_counts'])),
            })

            # Save per-(sub, ROI) matrices for later visualization
            safe_roi = (
                roi_name.replace(' ', '_').replace(',', '')
                        .replace('+', '').replace('(', '').replace(')', '')
                        .replace("'", '')
            )
            np.savez(
                os.path.join(matrices_dir, f"sub-mm{sub}_roi-{safe_roi}_similarity_cv.npz"),
                AA=AA, VV=VV, CC=CC,
                resid_matrix_cv=out['resid_matrix_cv'],
                resid_matrix_noI_cv=out['resid_matrix_noI_cv'],
                resid_matrix_noI_avg=out['resid_matrix_noI_avg'],
                cell_fold_counts=out['cell_fold_counts'],
                betas_per_fold=out['betas_per_fold'],
                betas_per_fold_noI=out['betas_per_fold_noI'],
            )

            t2 = time.time()
            print(
                f"sub {sub} | ROI {roi_name} | "
                f"CV: b_MM={out['beta_cv'][3]:+.4f}  "
                f"R²(A+V)={out['cv_r2_unimodal']:+.3f}  R²(full)={out['cv_r2']:+.3f}  "
                f"resid-MSPS(noI)={out['resid_msps_noI_cv']:+.4f} | {t2 - t1:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df['roi'] = df['roi'].replace({
        'Occipital Pole': 'OP',
        "Heschl's Gyrus (includes H1 and H2)": 'HG',
        "Superior Temporal Gyrus, anterior division": 'STG_A',
        "Superior Temporal Gyrus, posterior division": 'STG_P',
        "Temporal Pole": 'TP',
    })
    df.to_csv(results_csv_path, index=False)
    return df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sub', '--sub_id', type=str, required=True)
    p = parser.parse_args()

    subs = [p.sub_id]
    atlas = "ASHS"

    # Same ROI configuration as run_rsa_w_exclusion.py
    if atlas == "ASHS":
        rois = [
            "post_right_HPC_mask_T1", "ant_right_HPC_mask_T1",
            "post_left_HPC_mask_T1",  "ant_left_HPC_mask_T1",
            "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1",
            "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum",
            "HPC", "HPC",
            "Heschl's Gyrus (includes H1 and H2)",
            "Occipital Pole",
            "Superior Temporal Gyrus, posterior division",
            "Lateral Occipital Cortex, inferior division",
        ]
        rois_bool = [0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1,
                     2, 2, 2, 2]
        pos = ["", "", "", "", "", "",
               "combined", "combined", "combined", "combined",
               "combined", "combined", "combined", "combined",
               "left", "right",
               "", "", "", ""]

    nruns = 9
    norm = False
    apply_exclusion = True
    user = "or62"

    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed"
    exclusion_path = (
        f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/"
        "decoding_structs/greater_than_1_5_exclusion.csv"
    )

    results_dir = (
        f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/"
        f"RSA_structs/{atlas}/similarity_cv_multiple_regression"
    )
    matrices_dir = os.path.join(results_dir, "per_subject_matrices")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(matrices_dir, exist_ok=True)

    results_csv_path = os.path.join(
        results_dir,
        f"sub_{subs[0]}_atlas_{atlas}_exclusion_{apply_exclusion}_"
        "similarity_cv_multiple_regression_Results.csv",
    )

    run_cv_regression_analysis(
        rois=rois, rois_bool=rois_bool, pos=pos,
        subs=subs, nruns=nruns, norm=norm,
        preproc_dir=preproc_dir,
        exclusion_path=exclusion_path,
        apply_exclusion=apply_exclusion,
        atlas=atlas,
        results_csv_path=results_csv_path,
        matrices_dir=matrices_dir,
    )
