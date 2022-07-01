import pandas as pd
import matplotlib.cm

import DN_population_analysis.utils as utils


trial_dirs = utils.load_exp_dirs("../../recordings.txt")
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

results = pd.read_csv("output/Fig3/ball_rot_prediction_results.csv", index_col=0).reset_index(drop=True)
results = results.loc[(results["Variable"] == "turn_l") | (results["Variable"] == "turn_r"), :]
results = results.groupby(["Fly", "ROI", "Variable"]).mean().reset_index()
results["direction"] = 1
results.loc[results["Variable"] == "turn_l", "direction"] = -1
utils.generate_position_plots(trial_dirs, "output/Fig3", "turning_rsquared", results, "direction", "rsquared", cmap=matplotlib.cm.bwr, reduction_function=utils.pick_max_behaviour, figsize=(1.24, 0.6), kde=True)

results = pd.read_csv("output/Fig3/ball_rot_prediction_results.csv", index_col=0).reset_index(drop=True)
results = results.loc[results["Variable"] == "speed", :]
results["color_val"] = 1
results = results.groupby(["Fly", "ROI", "Variable"]).mean().reset_index()
utils.generate_position_plots(trial_dirs, "output/Fig3", "speed_rsquared", results, "color_val", "rsquared", cmap=matplotlib.cm.Set2, rsquared_thresh=0.02, kde=True)
