import numpy as np
import pandas as pd
import tqdm
import sklearn.linear_model

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, fictrac=False, angles=False, active=False, odor=True, joint_positions=False)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]
    fly_df = fly_df.reset_index(drop=True)
    
    matrix_results = pd.read_csv("output/Fig2/behavior_prediction_results.csv")
    matrix_results = matrix_results.loc[matrix_results["Context"] == "all", :]
    matrix_results = matrix_results.loc[~matrix_results["Variable"].isin(["turn_l", "turn_r", "ACV", "MSC", "all_odor"]), :]
    matrix_results = matrix_results.loc[matrix_results["rsquared"] > 0.05, :]
    matrix_results_fly = matrix_results.loc[matrix_results["Fly"] == f"{date}_{fly}", :]
    matrix_results_fly = matrix_results_fly.sort_values("rsquared", ascending=False)
    matrix_results_fly = matrix_results_fly.drop_duplicates("ROI", keep="first")
    matrix_results_fly = matrix_results_fly.loc[matrix_results_fly["Variable"].isin(["walking", "hind_grooming"]), :]
    rois = matrix_results_fly["ROI"].unique()

    for roi in tqdm.tqdm(rois):
        roi_df = fly_df.loc[fly_df["ROI"] == roi, :]
        endog = roi_df["dFF"].values

        baseline_exog = []
        for trial in roi_df["Trial"].unique():
            trial_mask = (roi_df["Trial"] == trial)
            trial_baseline = trial_mask.values.astype(float)
            baseline_exog.append(trial_baseline)
        baseline_exog = np.array(baseline_exog).transpose()

        walking_regressor = (roi_df["Behaviour"] == "walking").values.astype(float)
        hind_regressor = (roi_df["Behaviour"] == "hind_grooming").values.astype(float)
        walking_exog = [utils.convolve_with_crf(np.arange(len(walking_regressor)) / 100, walking_regressor, roi_df["Trial"]), ]
        hind_exog = [utils.convolve_with_crf(np.arange(len(hind_regressor)) / 100, hind_regressor, roi_df["Trial"]), ]
        walking_exog = np.array(walking_exog).transpose()
        hind_exog = np.array(hind_exog).transpose()

        trials = roi_df["Trial"].values
        sub_roi_df = roi_df.loc[roi_df["Behaviour"].isin(["walking", "hind_grooming"]), :]
        sub_roi_df = utils.balance_df(roi_df, "Behaviour", trials)
        beh_mask = roi_df.index.isin(sub_roi_df.index)
        baseline_exog = baseline_exog[beh_mask]
        walking_exog = walking_exog[beh_mask]
        hind_exog = hind_exog[beh_mask]
        endog = endog[beh_mask]
        trials = trials[beh_mask]
            
        full_exog = np.concatenate((baseline_exog, walking_exog, hind_exog), axis=1)
        full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * (walking_exog.shape[1] + hind_exog.shape[1])).transpose()
        
        for beh in ["walking", "hind_grooming"]:
            print(beh)

            # Remove variance cause by variable baseline
            lm = sklearn.linear_model.LinearRegression()
            lm.fit(baseline_exog, endog)
            endog = endog - lm.predict(baseline_exog)
            
            if beh == "walking":
                shuffled_beh_exog = np.copy(walking_exog)
                non_shuffled_beh_exog = hind_exog
            elif beh == "hind_grooming":
                shuffled_beh_exog = np.copy(hind_exog)
                non_shuffled_beh_exog = walking_exog
            else:
                raise ValueError(f"Unknown behaviour {beh}.")
            
            np.random.shuffle(shuffled_beh_exog)
            
            reduced_exog = np.concatenate((baseline_exog, shuffled_beh_exog, non_shuffled_beh_exog), axis=1)
            reduced_bounds = np.copy(full_bounds)

            prediction_plots_dir = "hind_prediction_plots"
            prediction_plots_dir = None
            results = results.append(utils.cv_rsquared(date, fly, beh, roi, full_exog, endog, reduced_exog, trials.copy(), full_bounds, reduced_bounds, one_model_per_trial=True, prediction_plots_dir=prediction_plots_dir))

results.to_csv("output/FigS2/predict_activity_walkingVShind.csv")
