import os

import numpy as np
import pandas as pd
import tqdm
import sklearn.linear_model

import utils2p

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, odor=True, fictrac=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]
    
    thresh = 0
    fly_df["turn_r"] = fly_df["turn"].clip(lower=thresh) - thresh
    fly_df["turn_l"] = (fly_df["turn"].clip(upper=thresh) - thresh) * -1
    fly_df["speed"] = fly_df["vel"].clip(lower=thresh) - thresh

    for beh in ["turn_r", "turn_l", "speed", "vel"]:
        if beh == "background":
            continue
        print(beh)
        for roi in tqdm.tqdm(fly_df["ROI"].unique()):
            roi_df = fly_df.loc[fly_df["ROI"] == roi, :]

            endog = roi_df[beh].values.astype(float)
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

            # Remove variance caused by variable baseline
            lm = sklearn.linear_model.LinearRegression()
            lm.fit(baseline_exog, endog)
            endog = endog - lm.predict(baseline_exog)

            if beh == "speed" or beh == "vel":
                walking_regressor = roi_df["Behaviour"] == "walking"
                walking_exog = utils.convolve_with_crf(np.arange(len(walking_regressor)) / 100, walking_regressor, roi_df["Trial"])[:, np.newaxis]
                full_exog = np.concatenate((baseline_exog, walking_exog, neural_exog), axis=1)
                full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * (walking_exog.shape[1] + neural_exog.shape[1])).transpose()
                reduced_exog = np.concatenate((baseline_exog, walking_exog, shuffled_neural_exog), axis=1)
            else:
                full_exog = np.concatenate((baseline_exog, neural_exog), axis=1)
                full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * neural_exog.shape[1]).transpose()
                reduced_exog = np.concatenate((baseline_exog, shuffled_neural_exog), axis=1)
            
            trials = roi_df["Trial"].values

            walking_mask = roi_df["Behaviour"] == "walking"
            full_exog = full_exog[walking_mask, :]
            endog = endog[walking_mask]
            reduced_exog = reduced_exog[walking_mask, :]
            trials = trials[walking_mask]
            
            results = results.append(utils.cv_rsquared(date, fly, beh, roi, full_exog, endog, reduced_exog, trials, full_bounds, full_bounds.copy(), one_model_per_trial=True))

    results.to_csv(f"output/Fig3/ball_rot_prediction_results.csv")

results.to_csv("output/Fig3/ball_rot_prediction_results.csv")
