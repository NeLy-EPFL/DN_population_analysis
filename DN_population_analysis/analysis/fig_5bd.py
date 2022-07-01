import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils2p.synchronization
import sklearn.linear_model

import DN_population_analysis.utils as utils


fly_rois = {210830: [3, 47, 58, 79],
            210910: [1, 27, 85, 66],
            211026: [7, 26, 53, 69],
            211027: [6, 16, 54, 70],
            211029: [4, 28, 46, 65],
           }

exp_dirs = utils.load_exp_dirs("../../recordings.txt")

plot_df = pd.DataFrame()

for fly_dir, trial_dirs in utils.group_by_fly(exp_dirs).items():
    for trial_dir in trial_dirs:
        date = utils.get_date(trial_dir)
        trial = utils.get_trial_number(trial_dir)
        trial_df = utils.load_fly_data([trial_dir,], dFF=True, antenna_touches=True, odor=True, active=True)
        trial_df = trial_df.loc[trial_df["Odor"].isin(["ACV", "MSC", "H2O"]), :]
        rois = fly_rois[date]
        trial_df = trial_df.loc[trial_df["ROI"].isin(rois[1:3]), :]
        trial_df = trial_df.drop(columns=["event_number", "event_based_index"])
        index_cols = [c for c in trial_df.columns if not c in ["ROI", "dFF", "active"]]
        trial_df = trial_df.pivot(index=index_cols, columns="ROI", values=["dFF", "active"]).reset_index()

        trial_df["Epoch indices l"], trial_df["Event number l"] = utils2p.synchronization.event_based_frame_indices(trial_df["L antenna touch"].values)
        trial_df["Epoch indices r"], trial_df["Event number r"] = utils2p.synchronization.event_based_frame_indices(trial_df["R antenna touch"].values)
        
        fig, axes = plt.subplots(5, 1, figsize=(3, 1), sharex=True)
        axes[2].plot(trial_df["Frame"].values / 100, (trial_df[("active", rois[1])] - trial_df[("active", rois[2])]).values)
        for i, side in enumerate(["l", "r"]):
            for epoch_number, epoch_df in trial_df.groupby(f"Event number {side}"):
                epoch_df = epoch_df.loc[epoch_df[f"Epoch indices {side}"] >= 0, :]
                start = epoch_df["Frame"].min() / 100
                stop = epoch_df["Frame"].max() / 100
                axes[i + 3].axvspan(start - 1, stop + 1, facecolor="black", alpha=0.5)
            t = trial_df["Frame"].values / 100
            if side == "l":
                y = trial_df[("dFF", rois[1])].values
                #axes[i].plot(trial_df["Frame"].values / 100, trial_df[("active", rois[1])].values * 2500)
                axes[i].plot(t, y)
                regressor = utils.convolve_with_crf(t, trial_df["L antenna touch"].values, trials=None)
                lm = sklearn.linear_model.LinearRegression()
                lm.fit(regressor[:, np.newaxis], y)
                prediction = lm.predict(regressor[:, np.newaxis])
                axes[i].plot(t, prediction)
            elif side == "r":
                y = trial_df[("dFF", rois[2])].values
                #axes[i].plot(t, trial_df[("active", rois[2])].values * 2500)
                axes[i].plot(t, y)
                regressor = utils.convolve_with_crf(t, trial_df["R antenna touch"].values, trials=None)
                lm = sklearn.linear_model.LinearRegression()
                lm.fit(regressor[:, np.newaxis], y)
                prediction = lm.predict(regressor[:, np.newaxis])
                axes[i].plot(t, prediction)
            else:
                raise NotImplemented
        plt.savefig(f"output/Fig5/panel_bd_{date}_{trial:03}.pdf")
        plt.close()
        quit()
#
#        if side == "l":
#            trial_df["Epoch indices"] , trial_df["Event number"] = utils2p.synchronization.event_based_frame_indices((trial_df["Assym antenna touch"] > 0).values)
#        elif side == "r":
#            trial_df["Epoch indices"] , trial_df["Event number"] = utils2p.synchronization.event_based_frame_indices((trial_df["Assym antenna touch"] < 0).values)
#        else:
#            raise NotImplemented
#        
#        max_roi1 = np.max(trial_df[rois[1]])
#        max_roi2 = np.max(trial_df[rois[2]])
#        for epoch, epoch_df in trial_df.groupby("Event number"):
#            if epoch == -1:
#                continue
#            print(epoch)
#            fig, axes = plt.subplots(2, 1, sharex=True)
#            axes[0].plot((epoch_df["Epoch indices"] >= 0).values * max_roi1)
#            axes[0].plot(epoch_df[rois[1]].values)
#            axes[0].set_ylim(-100, max_roi1 + 100)
#            axes[1].plot((epoch_df["Epoch indices"] >= 0).values * max_roi2)
#            axes[1].plot(epoch_df[rois[2]].values)
#            axes[1].set_ylim(-100, max_roi2 + 100)
#            plt.savefig(f"FigGrooming/assym_{side}/{date}_{trial:03}_{epoch}.pdf")
#            plt.close()
#
#        trial_df[rois[1]] = trial_df[rois[1]] / max_roi1
#        trial_df[rois[2]] = trial_df[rois[2]] / max_roi2
#        trial_df = trial_df.loc[trial_df["Event number"] >= 0, :]
#        trial_df = trial_df.loc[trial_df["Epoch indices"] >= 0, :]
#        trial_df = trial_df.rename(columns={rois[1]: 0, rois[2]: 1})
#
#        plot_df = plot_df.append(trial_df)
#plot_df.to_csv(f"FigGrooming/plot_df_{side}.csv")
#
#plot_df = pd.read_csv(f"FigGrooming/plot_df_{side}.csv", index_col=0)
#plot_df = plot_df.rename(columns={"0": 0, "1": 1})
#
#fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 7))
#for (date, trial, event_number), epoch_df in plot_df.groupby(["Date", "Trial", "Event number"]):
#    axes[0].plot(epoch_df[0].values - epoch_df[0].values[0], color="black", alpha=0.1)
#    axes[1].plot(epoch_df[1].values - epoch_df[1].values[0], color="black", alpha=0.1)
#    plot_df.loc[(plot_df["Date"] == date) & (plot_df["Trial"] == trial) & (plot_df["Event number"] == event_number), 0] = epoch_df[0].values - epoch_df[0].values[0]
#    plot_df.loc[(plot_df["Date"] == date) & (plot_df["Trial"] == trial) & (plot_df["Event number"] == event_number), 1] = epoch_df[1].values - epoch_df[1].values[0]
#mean_df = plot_df.groupby("Epoch indices").mean()
#axes[0].plot(mean_df[0].values, color="green")
#axes[1].plot(mean_df[1].values, color="green")
#plt.savefig(f"FigGrooming/assym_{side}/summary.pdf")
#plt.close()
