import os.path

import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import DN_population_analysis.utils as utils


def pick_max_corr(df):
    df = df.drop(columns=["Date", "Trial", "ROI"])
    return df.loc[df["corr"].idxmax(), :]

crosscorr_df = pd.DataFrame()

for trial_dir in utils.load_exp_dirs("../../recordings.txt"):
    date = utils.get_date(trial_dir)
    trial = utils.get_trial_number(trial_dir)
    print(date, trial)

    df = pd.read_pickle(os.path.join(trial_dir, "2p/roi_dFF_2p.pkl"))
    df = df.reset_index()

    denoised_df = df.loc[df["Source file"] == "denoised", :].pivot(columns="ROI", index="Frame", values="dFF").reset_index()
    raw_df = df.loc[df["Source file"] == "warped", :].pivot(columns="ROI", index="Frame", values="dFF").reset_index()
    raw_df = raw_df.loc[raw_df["Frame"].isin(denoised_df["Frame"]), :]
    denoised_df = denoised_df.loc[denoised_df["Frame"].isin(raw_df["Frame"]), :]

    for roi in df["ROI"].unique():
        print("\t", roi)
        lm = sklearn.linear_model.LinearRegression()
        lm.fit(denoised_df[roi].values[:, np.newaxis], raw_df[roi].values)
        fitted_denoised = lm.predict(denoised_df[roi].values[:, np.newaxis])

        if date == 210830 and trial == 1 and roi in (0, 2, 6):
            plt.figure(figsize=(3, 0.75))
            ax = plt.gca()
            ax.set_clip_on=False
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            x = np.arange(raw_df.shape[0]) / 16.27
            plt.plot(x, raw_df[roi].values, color="black")
            plt.plot(x, fitted_denoised, color="#02C12A")
            plt.ylabel("dF/F")
            plt.savefig(f"output/FigS1/panel_a_{date}_{trial:03}_{roi}_2p.pdf", transparent=True)
            plt.close()

        l = len(fitted_denoised)
        roi_df = pd.DataFrame()
        for lag in range(-32, 33):
            x = raw_df[roi].values
            y = denoised_df[roi].values #fitted_denoised
            if lag < 0:
                x = x[-lag:]
                y = y[:lag]
            if lag > 0:
                x = x[:-lag]
                y = y[lag:]
            corrcoef = np.corrcoef(x, y)[0, 1]
            lag_df = pd.DataFrame({"Date": [date,], "Trial": [trial,], "ROI": [roi,], "Lag": [lag,], "corr": [corrcoef,]})
            roi_df = roi_df.append(lag_df)
        crosscorr_df = crosscorr_df.append(roi_df)
    crosscorr_df.to_csv("output/FigS1/denoising_cross_corr.csv")

crosscorr_df = pd.read_csv("output/FigS1/denoising_cross_corr.csv", index_col=0).reset_index(drop=True)
crosscorr_df["Lag"] = crosscorr_df["Lag"] / 16.27
crosscorr_df = crosscorr_df.rename(columns={"Lag": "Lag (s)"})
max_crosscorr_df = crosscorr_df.groupby(["Date", "ROI"]).apply(pick_max_corr).reset_index()
crosscorr_df = crosscorr_df.rename(columns={"corr": "Pearson r"})

max_crosscorr_df["Sham"] = "Sham"

sns.set_theme(style="ticks", rc=utils.SEABORN_RC)

fig, axes = plt.subplots(2, 1, figsize=(1.5, 1.9), gridspec_kw={"height_ratios": [1, 2]}, sharex=True)
sns.boxplot(x="Lag (s)", y="Sham", data=max_crosscorr_df, showfliers=False, ax=axes[0])
sns.swarmplot(x="Lag (s)", y="Sham", data=max_crosscorr_df, hue="Date", size=1, ax=axes[0])
axes[0].legend([], [], frameon=False)
axes[0].axis("off")
axes[1].axvline(x=0, linestyle="--", alpha=0.5, color="black")
sns.lineplot(data=crosscorr_df, x="Lag (s)", y="Pearson r", ax=axes[1])
plt.savefig("output/FigS1/panel_b.pdf", transparent=True)
plt.close()
