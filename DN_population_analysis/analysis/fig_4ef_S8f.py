import pandas as pd

import DN_population_analysis.utils as utils


trial_dirs = sorted(utils.load_exp_dirs("../../recordings.txt"))
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

results = pd.read_csv("output/Fig4/odor_regression_results.csv", index_col=0).reset_index(drop=True)
results = results.loc[results["Variable"].isin(["ACV", "MSC"]), :]

utils.generate_position_plots(trial_dirs, "output/Fig4", "odor", results, "Variable", "rsquared", reduction_function=utils.pick_max_behaviour, vmax_size=0.35, rsquared_thresh=0.02)
