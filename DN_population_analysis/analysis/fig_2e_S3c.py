import pandas as pd

import DN_population_analysis.utils as utils


trial_dirs = utils.load_exp_dirs("../../recordings.txt")
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

results = pd.read_csv("output/Fig2/behavior_prediction_results.csv", index_col=0).reset_index(drop=True)
results = results.loc[results["Context"] == "all", :]
utils.generate_position_plots(trial_dirs, "output/Fig2", "behaviour", results, "Variable", "rsquared", reduction_function=utils.pick_max_behaviour, vmax_size=0.9)
