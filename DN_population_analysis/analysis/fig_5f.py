import numpy as np
import matplotlib.pyplot as plt

import DN_population_analysis.utils as utils


for trial_dir in utils.load_exp_dirs("../../R65D11_recordings.txt"):
    print(trial_dir)
    trial = utils.get_trial_number(trial_dir)

    df = utils.load_R65D11_data([trial_dir,])

    min_dff, max_dff = df["dFF"].min(), df["dFF"].max()
    
    behaviors = df.loc[df["ROI"] == 0, "Behaviour"]
    x = np.arange(len(behaviors)) / 30
    stimulus = df.loc[df["ROI"] == 0, "CO2"].values

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(3, 1))
    utils.behaviour_shading(axes[0], x, behaviors)
    utils.stimulus_shading(axes[1], x, stimulus)
    axes[2].plot(x, df.loc[df["ROI"] == 0, "dFF"].values)
    axes[2].set_ylim(min_dff, max_dff)
    axes[3].plot(x, df.loc[df["ROI"] == 1, "dFF"].values)
    axes[3].set_ylim(min_dff, max_dff)
    plt.savefig(f"output/Fig5/panel_f_trial_{trial:03}.pdf")
    plt.close()
