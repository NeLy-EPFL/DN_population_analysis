import os.path

import numpy as np
import pandas as pd
import sklearn.linear_model

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()
masked_results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    genotype = utils.get_genotype(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, fictrac=True, odor=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]
    
    rois = fly_df["ROI"].unique()
    index_cols = ["Trial", "Frame", "Behaviour", "conv_vel", "conv_turn", "vel", "turn"]
    roi_df = fly_df.pivot(index=index_cols, columns="ROI", values="dFF").sort_index()
    roi_df = roi_df.reset_index()
    behaviours = roi_df["Behaviour"].unique()

    baseline_exog = []
    neural_exog = []
    shuffled_neural_exog = []
    behaviour_exog = []

    beh_regressor_names = []
    roi_regressor_names = []

    walking_mask = np.zeros(roi_df.shape[0], dtype=bool)

    for trial in roi_df["Trial"].unique():

        trial_mask = (roi_df["Trial"] == trial)

        trial_baseline = trial_mask.values.astype(float)
        baseline_exog.append(trial_baseline)

        walking_mask[trial_mask] = (roi_df.loc[trial_mask, "Behaviour"] == "walking")

        for beh in behaviours:
            binary_beh = np.zeros(roi_df.shape[0])
            binary_beh[trial_mask] = utils.convolve_with_crf(np.arange(roi_df.shape[0]) / 100, roi_df.loc[trial_mask, "Behaviour"] == beh, roi_df.loc[trial_mask, "Trial"])
            behaviour_exog.append(binary_beh)
            beh_regressor_names.append(f"Trial_{trial}_{beh}")

        dFF = np.zeros((roi_df.shape[0], len(rois)))
        dFF[trial_mask] = roi_df.loc[trial_mask, rois].values
        neural_exog.append(dFF)
        
        shuffled_dFF = np.zeros((roi_df.shape[0], len(rois)))
        tmp = roi_df.loc[trial_mask, rois].values
        np.random.shuffle(tmp)
        shuffled_dFF[trial_mask] = tmp
        shuffled_neural_exog.append(shuffled_dFF)

        roi_regressor_names.extend([f"Trial_{trial}_ROI_{roi}" for roi in rois])


    baseline_exog = np.array(baseline_exog).transpose()
    neural_exog = np.concatenate(neural_exog, axis=1)
    shuffled_neural_exog = np.concatenate(shuffled_neural_exog, axis=1)
    behaviour_exog = np.array(behaviour_exog).transpose()

    full_exog = np.concatenate((baseline_exog, behaviour_exog, neural_exog), axis=1)
    reduced_exog = np.concatenate((baseline_exog, behaviour_exog, shuffled_neural_exog), axis=1)

    baseline_regressor_names = [f"Baseline {i}" for i in range(baseline_exog.shape[1])]
    regressor_names = baseline_regressor_names + beh_regressor_names + roi_regressor_names

    masked_full_exog = full_exog[walking_mask, :]
    masked_reduced_exog = reduced_exog[walking_mask, :]

    for rot in ("conv_turn", "conv_vel", "turn", "vel"):
        endog = roi_df[rot].values.astype(float)

        masked_endog = endog[walking_mask]

        ## Remove variance cause by variable baseline
        #lm = sklearn.linear_model.LinearRegression()
        #lm.fit(baseline_exog[walking_mask], masked_endog)
        #masked_endog = masked_endog - lm.predict(baseline_exog[walking_mask])

        trials = roi_df["Trial"].values
        masked_trials = trials[walking_mask]

        prediction_plots_dir = "output/Fig3/prediction_plots_population"
        os.makedirs(prediction_plots_dir, exist_ok=True)
        results = results.append(utils.cv_rsquared(date, fly, rot, 0, full_exog, endog, reduced_exog, trials, None, None, one_model_per_trial=True, prediction_plots_dir=prediction_plots_dir))
        masked_results = masked_results.append(utils.cv_rsquared(date, fly, rot, 0, masked_full_exog, masked_endog, masked_reduced_exog, masked_trials, None, None, one_model_per_trial=True))

results.to_csv("output/Fig3/ball_rot_prediction_population_results.csv")
masked_results.to_csv("output/Fig3/ball_rot_prediction_population_masked_results.csv")
