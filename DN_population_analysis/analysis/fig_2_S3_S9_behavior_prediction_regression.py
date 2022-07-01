import os

import numpy as np
import pandas as pd
import tqdm

import utils2p

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, odor=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]

    for beh in list(fly_df["Behaviour"].unique()):
        if beh == "background":
            continue
        print(beh)
        for roi in tqdm.tqdm(fly_df["ROI"].unique()):
            roi_df = fly_df.loc[fly_df["ROI"] == roi, :]

            endog = (roi_df["Behaviour"] == beh).values.astype(float)
            endog = utils.convolve_with_crf(np.arange(len(endog)) / 100, endog, roi_df["Trial"])

            baseline_exog = []
            neural_exog = []
            shuffled_neural_exog = []
            bounds = []

            for trial in roi_df["Trial"].unique():

                trial_mask = (roi_df["Trial"] == trial)

                trial_baseline = trial_mask.values.astype(float)
                baseline_exog.append(trial_baseline)
                bounds.append([np.NINF, np.inf])

                dFF = np.zeros(roi_df.shape[0])
                dFF[trial_mask] = roi_df.loc[trial_mask, "dFF"].values
                neural_exog.append(dFF)
                
                shuffled_dFF = np.zeros(roi_df.shape[0])
                trial_dFF = roi_df.loc[trial_mask, "dFF"].values
                np.random.shuffle(trial_dFF)
                shuffled_dFF[trial_mask] = trial_dFF
                shuffled_neural_exog.append(shuffled_dFF)

            baseline_exog = np.array(baseline_exog).transpose()
            neural_exog = np.array(neural_exog).transpose()
            shuffled_neural_exog = np.array(shuffled_neural_exog).transpose()

            full_exog = np.concatenate((baseline_exog, neural_exog), axis=1)
            full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * neural_exog.shape[1]).transpose()
            reduced_exog = np.concatenate((baseline_exog, shuffled_neural_exog), axis=1)
            
            trials = roi_df["Trial"].values

            for context in ("all", "H2O", "ACV", "MSC",):
                if context == "all":
                    context_mask = roi_df["Odor"].isin(["ACV", "MSC", "H2O"]).values
                    n_subsets = 1
                else:
                    context_mask = (roi_df["Odor"] == context).values
                    block_size = roi_df["Odor"].value_counts().min()
                    n_subsets = roi_df["Odor"].value_counts()[context] // block_size

                event_frame_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(context_mask)
                event_lengths = {}
                for event in np.unique(event_numbers):
                    event_lengths[event] = np.max(event_frame_indices[event_numbers == event])

                for subset in range(n_subsets):
                    if n_subsets == 1:
                        subset_mask = context_mask
                    else:
                        subset_mask = np.zeros_like(context_mask)
                        for event in np.unique(event_numbers):
                            if event == -1:
                                continue
                            l = event_lengths[event] // n_subsets
                            subset_mask[(event_numbers == event) &
                                        (event_frame_indices >= subset * l) &
                                        (event_frame_indices < (subset + 1) * l)] = True

                    subset_full_exog = np.copy(full_exog[subset_mask]) 
                    subset_reduced_exog = np.copy(reduced_exog[subset_mask]) 
                    subset_endog = np.copy(endog[subset_mask]) 
                    subset_trials = np.copy(trials[subset_mask]) 

                    prediction_plots_dir = f"output/Fig2/behavior_prediction_traces/prediction_plots_dir_{context}"
                    os.makedirs(prediction_plots_dir, exist_ok=True)
                    #prediction_plots_dir = None
                    sub_results = utils.cv_rsquared(date, fly, beh, roi, subset_full_exog, subset_endog, subset_reduced_exog, subset_trials, full_bounds, full_bounds.copy(), one_model_per_trial=True, prediction_plots_dir=prediction_plots_dir)
                    sub_results["Context"] = context
                    sub_results["Subset"] = subset
                    results = pd.concat([results, sub_results], axis="index")

    results.to_csv(f"output/Fig2/behavior_prediction_results.csv")

results.to_csv("output/Fig2/behavior_prediction_results.csv")
