import os.path

import pandas as pd
import numpy as np

import DN_population_analysis.utils as utils


trial_dirs = utils.load_exp_dirs("../../recordings.txt")
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

results = pd.read_csv("output/Fig2/behavior_prediction_results.csv", index_col=0).reset_index(drop=True)
results = results.loc[results["Context"] == "all", :]
results = results.loc[results["Variable"].isin(utils.BEHAVIOUR_CLASSES), :]
rsquared_thresh = 0.05

results = results.groupby(["Fly", "ROI", "Variable"]).mean().reset_index()
results = results.loc[results["rsquared"] > rsquared_thresh, :]
results = results.groupby(["Fly", "ROI"]).apply(utils.pick_max_behaviour).reset_index()
for beh in ["walking", "head_grooming"]:
    x = []
    y = []
    for fly, fly_df in results.groupby("Fly"):
        date, fly_number = fly.split("_")
        fly_number = int(fly_number)
        trial_dir = [d for d in trial_dirs if utils.get_fly_number(d) == fly_number][0]

        beh_rois = fly_df.loc[fly_df["Variable"] == beh, "ROI"].values.astype(int)

        roi_centers = np.loadtxt(os.path.join(trial_dir, "2p/roi_centers.txt"))
        roi_centers = roi_centers[beh_rois]

        x.append(roi_centers[:, 0])
        y.append(roi_centers[:, 1])
    x = np.concatenate(x)
    y = np.concatenate(y)
    if len(x) == 1:
        continue
    utils.plot_kde_2D(x, y, f"output/Fig2/kde2d_{beh}.pdf")
