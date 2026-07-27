#!/gpfs/milgram/project/turk-browne/or62/conda_envs/myenv_multimem/bin/python
"""
Split-half RSA control analysis (reviewer comment 2, PLoS Biol revision).

Logic
-----
The reviewer asked us to verify the retrieval/imagery control with a simple
split-half: estimate the multivariate effects separately in the FIRST and
SECOND halves of the experiment and ask whether they strengthen with repeated
exposure (i.e. larger in the second half).

This script reuses the EXACT same RSA machinery as the main analysis
(run_rsa_w_exclusion.py). The only change is that, before building each RSA
matrix, we restrict the leave-one-run-out cross-validation to a subset of runs.
Three splits are computed in a single pass:

    first_half           -> runs 1-4   (4 runs)
    second_half          -> runs 5-9   (5 runs)  original split (4 vs 5)
    second_half_balanced -> runs 6-9   (4 runs)  balanced 4 vs 4 (drops run 5)

Compare first_half vs second_half for the original analysis, and first_half vs
second_half_balanced for the precision-matched version (equal run counts per
half -> matched estimation noise, so the larger half can't clear the
against-zero threshold more easily simply by being less noisy). mm03 (run 7) and
mm26 (run 8) lose a run to whole-run exclusion, so their balanced second half is
3 runs (4 vs 3); all other subjects are a clean 4 vs 4.

We do this by filtering `condlist` (and the row-aligned `masked_betas`) to the
runs in each half, then handing the restricted data to the unchanged
`get_rsa_matrix()` and `cross_modal()`. Because everything downstream is
identical to the main pipeline, the per-half numbers are directly comparable to
your main results.

Output
------
One CSV per subject (same columns as the main RSA output, plus a `split`
column):
    sub, cond, roi, corr, z_similarity, type, split

From this you compute, within each split:
    multisensory facilitation (posterior HPC) = MSPS(C-C) - MSPS(V-V)
    crossmodal transfer       (anterior HPC)  = MSPS(V-A)          [diag - off_diag]
where MSPS = (diagonal z) - (off-diagonal z). Each condition contributes two
rows here (type == 'diag' and type == 'off_diag'); MSPS is their difference.

Then run your usual group-level paired test comparing first vs. second half.

Run (one subject per job, as with the main pipeline):
    python rsa_splithalf.py -sub 01
"""

import warnings
import sys
import os
import time
import argparse

import numpy as np
import pandas as pd

# Suppress warnings (match main pipeline)
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import functions from the RSA script. This works whether rsa_splithalf.py is
# co-located in RSA/ (as in the split-half joblist) or run from a sibling dir
# like the searchlight script. We add the script's own dir and the parent, then
# try the co-located import first, falling back to the RSA.<module> package form.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
try:
    from run_rsa_w_exclusion import (
        load_glm_single,
        get_labels,
        extract_betas,
        get_rsa_matrix,
        cross_modal,
    )
except ImportError:
    from RSA.run_rsa_w_exclusion import (
        load_glm_single,
        get_labels,
        extract_betas,
        get_rsa_matrix,
        cross_modal,
    )


# Three run-splits, all computed in one pass so the full set is available:
#   first_half            runs 1-4   (4 runs)
#   second_half           runs 5-9   (5 runs)  -- ORIGINAL split (unbalanced 4 vs 5)
#   second_half_balanced  runs 6-9   (4 runs)  -- BALANCED 4 vs 4 with first_half
#                                                  (drops the middle run, run 5)
# Use first_half vs second_half for the original comparison, and
# first_half vs second_half_balanced for the precision-matched (equal-run-count)
# comparison. NOTE: mm03 (run 7) and mm26 (run 8) lose a run to whole-run
# exclusion, so their balanced second half is 3 runs (4 vs 3); all other
# subjects are a clean 4 vs 4.
SPLITS = {
    "first_half": [1, 2, 3, 4],
    "second_half": [5, 6, 7, 8, 9],
    "second_half_balanced": [6, 7, 8, 9],
}


def restrict_to_runs(condlist, masked_betas, run_subset):
    """Filter condlist and its row-aligned betas to a subset of runs.

    `masked_betas` rows correspond 1:1 (in order) to `condlist` rows, so the
    same boolean mask keeps them aligned. After filtering, get_rsa_matrix()
    derives `runs = condlist['runs'].unique()` from the subset, so leave-one-
    run-out cross-validation happens *within* the half.
    """
    keep = condlist["runs"].isin(run_subset).to_numpy()
    condlist_sub = condlist[keep].reset_index(drop=True)
    masked_betas_sub = masked_betas[keep]
    return condlist_sub, masked_betas_sub


def run_split_half(preproc_dir, sub, rois, rois_bool, pos, conditions, nruns,
                   norm, fisher, atlas, exclusion_path, apply_exclusion):
    """Compute cross-validated RSA within each run-split for one subject."""

    # --- load betas once (with the same exclusion logic as run_rsa) ---
    if apply_exclusion:
        exclusion_df = pd.read_csv(exclusion_path)
        if f"mm{sub}" in exclusion_df["subs"].values:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}_excluded/")
        else:
            betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")
    else:
        betas = load_glm_single(f"{preproc_dir}/glm_single_results_mm{sub}/")

    # --- labels once (identical to main pipeline) ---
    # NOTE on exclusion: get_labels() has ALREADY applied both forms of
    # exclusion here. (1) Whole-run exclusion: any run with >=3 flagged trials
    # is dropped, so it never appears in condlist['runs']. (2) Trial-level
    # exclusion: flagged trials in surviving runs are relabeled 'exclude_*' so
    # they don't match any S{n}_{cond} pattern. We therefore subset on top of
    # already-cleaned data; no exclusion logic is duplicated or bypassed.
    condlist = get_labels(f"{preproc_dir}/sub-mm{sub}/func/", sub,
                          exclusion_path, apply_exclusion, nruns)

    # --- diagnostic: how many of each split's runs survive exclusion? ---
    surviving = sorted(condlist["runs"].unique().tolist())
    print(f"[sub {sub}] surviving runs after exclusion: {surviving}")
    for split_label, run_subset in SPLITS.items():
        present = [r for r in surviving if r in run_subset]
        msg = (f"[sub {sub}]   {split_label}: runs {present} "
               f"(n={len(present)}/{len(run_subset)})")
        if len(present) < len(run_subset):
            msg += "  <<< NOTE: a run was dropped by whole-run exclusion"
        print(msg)

    rows = []  # accumulate result rows

    for r in range(len(rois)):
        roi = rois[r]

        # Mask betas for this ROI once (full run set). NOTE: norm=False in the
        # main config, so no normalisation happens here. If you ever set
        # norm=True, normalisation should be redone *within* each split for a
        # fully leak-free estimate (see README note in the chat).
        masked_betas = extract_betas(sub, preproc_dir, roi, betas, condlist,
                                     rois_bool[r], pos[r], norm, atlas)

        # CRITICAL alignment check: get_rsa_matrix assumes row i of betas
        # corresponds to row i of condlist. If this ever fails, every result
        # below would be silently wrong, so we assert it explicitly.
        assert masked_betas.shape[0] == len(condlist), (
            f"Beta/label misalignment for sub {sub}, roi {roi}: "
            f"{masked_betas.shape[0]} beta rows vs {len(condlist)} labels"
        )

        roi_name = f"{pos[r]}_{roi}" if roi == "HPC" else roi

        for split_label, run_subset in SPLITS.items():
            condlist_sub, masked_betas_sub = restrict_to_runs(
                condlist, masked_betas, run_subset
            )

            for cond in conditions:
                rsa_matrix = get_rsa_matrix(cond.split("-"), condlist_sub,
                                            masked_betas_sub)
                rsa_matrix = rsa_matrix.to_numpy()

                # cross_modal returns diagonal and off-diagonal (z and r).
                metrics, metrics_z, types = cross_modal(rsa_matrix, cond, fisher)

                for m in range(len(metrics)):
                    rows.append([sub, cond, roi_name,
                                 metrics[m], metrics_z[m], types[m],
                                 split_label])

    df = pd.DataFrame(
        rows,
        columns=["sub", "cond", "roi", "corr", "z_similarity", "type", "split"],
    )

    # Same sensory-ROI relabeling as the main pipeline.
    df["roi"] = df["roi"].replace({
        "Occipital Pole": "OP",
        "Heschl's Gyrus (includes H1 and H2)": "HG",
        "Superior Temporal Gyrus, anterior division": "STG_A",
        "Superior Temporal Gyrus, posterior division": "STG_P",
        "Temporal Pole": "TP",
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--sub_id", type=str)
    p = parser.parse_args()

    sub = p.sub_id

    # ---- configuration (mirrors run_rsa_w_exclusion.py __main__) ----
    atlas = "ASHS"

    rois = ["post_right_HPC_mask_T1", "ant_right_HPC_mask_T1",
            "post_left_HPC_mask_T1", "ant_left_HPC_mask_T1",
            "post_combined_HPC_mask_T1", "ant_combined_HPC_mask_T1",
            "HPC", "CA1", "CA2+3", "DG", "EC", "PHC", "PRC", "Subiculum",
            "HPC", "HPC",
            "Heschl's Gyrus (includes H1 and H2)", "Occipital Pole",
            "Superior Temporal Gyrus, posterior division",
            "Lateral Occipital Cortex, inferior division"]
    rois_bool = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    pos = ["", "", "", "", "", "", "combined", "combined", "combined",
           "combined", "combined", "combined", "combined", "combined",
           "left", "right", "", "", "", ""]

    # Conditions needed for the reviewer effects (run on the full ROI list):
    #   facilitation -> C-C vs V-V (and A-A for congruent>auditory)
    #   crossmodal   -> V-A
    conditions = ['V-V', 'A-A', 'C-C', 'V-A']

    nruns = 9
    norm = False
    fisher = True
    apply_exclusion = True
    user = "or62"
    preproc_dir = f"/gpfs/milgram/scratch60/turk-browne/{user}/sandbox/preprocessed"
    exclusion_path = "/gpfs/milgram/scratch60/turk-browne/or62/sandbox/decoding_structs/greater_than_1_5_exclusion.csv"

    results_path = f"/gpfs/milgram/scratch60/turk-browne/or62/sandbox/RSA_structs/split_half/{atlas}/"
    os.makedirs(results_path, exist_ok=True)
    results_file_name = (
        f"sub_{sub}_atlas_{atlas}_fisher_{fisher}_norm_{norm}"
        f"_exclusion_{apply_exclusion}_RSA_Results_splithalf_glmC.csv"
    )

    t1 = time.time()
    df = run_split_half(
        preproc_dir=preproc_dir, sub=sub, rois=rois, rois_bool=rois_bool,
        pos=pos, conditions=conditions, nruns=nruns, norm=norm, fisher=fisher,
        atlas=atlas, exclusion_path=exclusion_path, apply_exclusion=apply_exclusion,
    )
    df.to_csv(results_path + results_file_name, index=False)
    t2 = time.time()

    # ---- per-subject sanity summary for the two reviewer effects ----
    # MSPS = (diagonal z) - (off-diagonal z), per condition/roi/split.
    def msps(cond, roi_name, split_label):
        sel = df[(df["cond"] == cond) & (df["roi"] == roi_name)
                 & (df["split"] == split_label)]
        try:
            diag = sel.loc[sel["type"] == "diag", "z_similarity"].iloc[0]
            off = sel.loc[sel["type"] == "off_diag", "z_similarity"].iloc[0]
        except IndexError:
            return np.nan
        return diag - off

    print(f"\n[sub {sub}] effect summary (z-space MSPS); NaN => subject drops "
          f"from that effect/half in the group test):")
    for split_label in SPLITS:
        print(f"  [{split_label}]")
        for roi_name in df["roi"].unique():
            facil = msps("C-C", roi_name, split_label) \
                - msps("V-V", roi_name, split_label)
            xmod = msps("V-A", roi_name, split_label)
            print(f"    {roi_name:<32s} facilitation (C-C minus V-V) = "
                  f"{facil: .4f} | crossmodal transfer (V-A) = {xmod: .4f}")

    print(f"\nFinished split-half RSA for sub {sub} in {t2 - t1:.1f}s -> "
          f"{results_path + results_file_name}", flush=True)