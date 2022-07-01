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
    matrix_results_fly = matrix_results_fly.loc[matrix_results_fly["Variable"].isin(["head_grooming", "foreleg_grooming"]), :]
    rois = matrix_results_fly["ROI"].unique()

    for roi in tqdm.tqdm(rois):
        roi_df = fly_df.loc[fly_df["ROI"] == roi, :]
        print(roi_df.shape)
        endog = roi_df["dFF"].values

        baseline_exog = []
        for trial in roi_df["Trial"].unique():
            trial_mask = (roi_df["Trial"] == trial)
            trial_baseline = trial_mask.values.astype(float)
            baseline_exog.append(trial_baseline)
        baseline_exog = np.array(baseline_exog).transpose()

        head_regressor = (roi_df["Behaviour"] == "head_grooming").values.astype(float)
        foreleg_regressor = (roi_df["Behaviour"] == "foreleg_grooming").values.astype(float)
        head_exog = [utils.convolve_with_crf(np.arange(len(head_regressor)) / 100, head_regressor, roi_df["Trial"]), ]
        foreleg_exog = [utils.convolve_with_crf(np.arange(len(foreleg_regressor)) / 100, foreleg_regressor, roi_df["Trial"]), ]
        head_exog = np.array(head_exog).transpose()
        foreleg_exog = np.array(foreleg_exog).transpose()

        trials = roi_df["Trial"].values
        sub_roi_df = roi_df.loc[roi_df["Behaviour"].isin(["walking", "hind_grooming"]), :]
        sub_roi_df = utils.balance_df(sub_roi_df, "Behaviour", trials)
        beh_mask = roi_df.index.isin(sub_roi_df.index)
        beh_mask = roi_df["Behaviour"].isin(["head_grooming", "foreleg_grooming"]).values
        baseline_exog = baseline_exog[beh_mask]
        head_exog = head_exog[beh_mask]
        foreleg_exog = foreleg_exog[beh_mask]
        endog = endog[beh_mask]
        trials = trials[beh_mask]
            
        full_exog = np.concatenate((baseline_exog, head_exog, foreleg_exog), axis=1)
        full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * (head_exog.shape[1] + foreleg_exog.shape[1])).transpose()
        
        for beh in ["head_grooming", "foreleg_grooming"]:
            print(beh)

            # Remove variance cause by variable baseline
            lm = sklearn.linear_model.LinearRegression()
            lm.fit(baseline_exog, endog)
            endog = endog - lm.predict(baseline_exog)
            
            if beh == "head_grooming":
                shuffled_beh_exog = np.copy(head_exog)
                non_shuffled_beh_exog = foreleg_exog
            elif beh == "foreleg_grooming":
                shuffled_beh_exog = np.copy(foreleg_exog)
                non_shuffled_beh_exog = head_exog
            else:
                raise ValueError(f"Unknown behaviour {beh}.")
            
            np.random.shuffle(shuffled_beh_exog)
            
            reduced_exog = np.concatenate((baseline_exog, shuffled_beh_exog, non_shuffled_beh_exog), axis=1)
            reduced_bounds = np.copy(full_bounds)

            prediction_plots_dir = "output/FigS2/grooming_prediction_plots"
            prediction_plots_dir = None
            results = results.append(utils.cv_rsquared(date, fly, beh, roi, full_exog, endog, reduced_exog, trials, full_bounds, reduced_bounds, one_model_per_trial=True, prediction_plots_dir=prediction_plots_dir))

results.to_csv("output/FigS2/predict_activity_grooming.csv")
