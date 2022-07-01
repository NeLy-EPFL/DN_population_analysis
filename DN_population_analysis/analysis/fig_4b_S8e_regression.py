import os

import numpy as np
import pandas as pd
import tqdm

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, fictrac=True, angles=False, active=False, odor=True, joint_positions=False)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]

    thresh = 0
    fly_df["turn_l"] = fly_df["turn"].clip(lower=thresh) - thresh
    fly_df["turn_r"] = (fly_df["turn"].clip(upper=thresh) - thresh) * -1

    for roi in tqdm.tqdm(fly_df["ROI"].unique()):
        roi_df = fly_df.loc[fly_df["ROI"] == roi, :]
        endog = roi_df["dFF"].values

        behaviours = list(fly_df["Behaviour"].unique())

        baseline_exog = []
        behaviour_exog = []
        for trial in roi_df["Trial"].unique():
            trial_mask = (roi_df["Trial"] == trial)
            trial_baseline = trial_mask.values.astype(float)
            baseline_exog.append(trial_baseline)
        for beh in behaviours + ["turn_l", "turn_r"]:
            if beh == "background":
                continue
            if beh in behaviours:
                regressor = (roi_df["Behaviour"] == beh).values.astype(float)
            else:
                regressor = roi_df[beh].values.astype(float)
            regressor =  utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"])
            behaviour_exog.append(regressor)
        baseline_exog = np.array(baseline_exog).transpose()
        behaviour_exog = np.array(behaviour_exog).transpose()

        for odor in ["ACV", "MSC", "all_odor"]:
            print(odor)
            if odor in ["ACV", "MSC"]:
                regressor = (roi_df["Odor"] == odor).values.astype(float)
            else:
                regressor = roi_df["Odor"].isin(["ACV", "MSC"]).values.astype(float)
            odor_exog = [utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"]), ]

            odor_exog = np.array(odor_exog).transpose()

            full_exog = np.concatenate((baseline_exog, behaviour_exog, odor_exog), axis=1)
            full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * behaviour_exog.shape[1] + [[0, np.inf],]).transpose()
            shuffled_odor_exog = np.copy(odor_exog)
            np.random.shuffle(shuffled_odor_exog)
            reduced_exog = np.concatenate((baseline_exog, behaviour_exog, shuffled_odor_exog), axis=1)
            reduced_bounds = np.copy(full_bounds)
            
            trials = roi_df["Trial"].values

            prediction_plots_dir = "output/Fig4/odor_prediction_plots"
            os.makedirs(prediction_plots_dir, exist_ok=True)
            #prediction_plots_dir = None
            results = results.append(utils.cv_rsquared(date, fly, odor, roi, full_exog, endog, reduced_exog, trials, full_bounds, reduced_bounds, one_model_per_trial=True, prediction_plots_dir=prediction_plots_dir))

    results.to_csv("output/Fig4/odor_regression_results.csv")

results.to_csv("output/Fig4/odor_regression_results.csv")
