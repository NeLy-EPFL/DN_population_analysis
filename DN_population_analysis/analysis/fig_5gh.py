import matplotlib.pyplot as plt
import utils2p.synchronization

import DN_population_analysis.utils as utils


trial_dirs = utils.load_exp_dirs("../../R65D11_recordings.txt")

df = utils.load_R65D11_data(trial_dirs, moving_average=0)
for trial in range(1, 9):
    event_based_indices, event_number = utils2p.synchronization.event_based_frame_indices(
            df.loc[(df["Trial"] == trial) & (df["ROI"] == 0), "CO2"].values
            )
    for roi in range(2):
        df.loc[(df["Trial"] == trial) & (df["ROI"] == roi), "Event_based_indices"] = event_based_indices
        df.loc[(df["Trial"] == trial) & (df["ROI"] == roi), "Event_number"] = event_number

df = df.loc[df["Event_number"] >= 0, :]
df = df.loc[~df["Behaviour"].isin(["head_grooming", "foreleg_grooming"]), :]
df = df.loc[df["Event_based_indices"] > -20, :]

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(2.5, 1))
for i in range(2):
    axes[i].axvline(0, linestyle="--", color="black", alpha=0.5)

for (trial, roi, event_number), event_df in df.groupby(["Trial", "ROI", "Event_number"]):
    axes[roi].plot(event_df["Event_based_indices"].values / 100,
                   event_df["dFF"].values,
                   color="black",
                   alpha=0.2,
                   linewidth=0.5,
                  )
mean_df = df.groupby(["ROI", "Event_based_indices"]).mean().reset_index()
for roi, roi_mean_df in mean_df.groupby("ROI"):
    axes[roi].plot(roi_mean_df["Event_based_indices"].values / 100,
                   roi_mean_df["dFF"].values,
                   color="green",
                   linewidth=0.75,
                  )

plt.savefig("output/Fig5/panels_gh.pdf")
plt.close()
