import pandas as pd
import numpy as np
import matplotlib.cm

import DN_population_analysis.utils as utils


trial_dirs = sorted(utils.load_exp_dirs("../../recordings.txt"))
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

results = pd.read_csv("output/Fig2/principal_components.csv")
results = results.pivot(index=["Date", "Genotype", "Fly", "ROI"], columns="PC", values="Loading").reset_index()
results["Fly"] = results["Date"].map(str) + "_" + results["Fly"].map(str)

roi_order = pd.read_csv("output/Fig2/order_behaviour_encoding.csv")
results = results.merge(roi_order, on=["Fly", "ROI"])

results["l"] = np.sqrt(results[0] ** 2 + results[1] ** 2) + 0.1
results["Color"] = 1
utils.generate_position_plots(trial_dirs, "output/FigS5", "pc_loadings_numbers", results, "Color", "l", numbers=True, cmap=matplotlib.cm.viridis_r, vmax_size=1.1, percent=False)
