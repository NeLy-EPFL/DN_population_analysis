import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import DN_population_analysis.utils as utils


df = pd.read_csv("output/Fig3/ball_rot_prediction_results.csv")
df = df.loc[df["Variable"].isin(["turn_r", "turn_l", "speed", "vel"]), :]
df = df.groupby(["Variable", "Fly", "ROI"]).mean().reset_index()
df["rsquared"] = df["rsquared"] * 100
    
matrix_results = pd.read_csv("output/Fig2/behavior_prediction_results.csv")
matrix_results = matrix_results.loc[matrix_results["Context"] == "all", :]
matrix_results = matrix_results.groupby(["Fly", "Variable", "ROI"]).mean().reset_index()
matrix_results.loc[matrix_results["rsquared"] < 0.05, "Variable"] = "undefined"

plt.figure(figsize=(1.5, 1.5))
color_r = np.array([1, 0, 0, 0.3])
color_l = np.array([0, 0, 1, 0.3])
mean_r = np.zeros(100)
mean_l = np.zeros(100)
x_mean = np.linspace(0, 100, 100)
for fly, fly_df in df.groupby("Fly"):
    #date, fly_number = fly.split("_")
    #beh_rois = utils.get_beh_rois("output/Fig2/behavior_prediction_results.csv", date, fly_number)
    #walking_rois = beh_rois.loc[beh_rois["Behaviour"] == "walking", "ROI"]
    walking_rois = matrix_results.loc[(matrix_results["Fly"] == fly) & (matrix_results["Variable"] == "walking"), "ROI"].unique()
    fly_df = fly_df.loc[fly_df["ROI"].isin(walking_rois), :]

    y_l = np.sort(fly_df.loc[fly_df["Variable"] == "turn_l", "rsquared"].values)[::-1]
    y_r = np.sort(fly_df.loc[fly_df["Variable"] == "turn_r", "rsquared"].values)[::-1]
    x = np.linspace(0, 100, len(y_l))
    plt.plot(x, y_l, color=np.copy(color_l))
    plt.plot(x, y_r, color=np.copy(color_r))
    mean_r = mean_r + np.interp(x_mean, x, y_r)
    mean_l = mean_l + np.interp(x_mean, x, y_l)
mean_r = mean_r / 5
mean_l = mean_l / 5
color_r[3] = 1
color_l[3] = 1
plt.ylim(0, 85)
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.plot(x_mean, mean_r, color=color_r)
plt.plot(x_mean, mean_l, color=color_l)
plt.xlabel("% walking ROIs (neurons)")
plt.ylabel("UEV (%)")
plt.savefig("output/Fig3/panel_e.pdf")
plt.close()

plt.figure(figsize=(1.5, 1.5))
color = np.array([0.4, 0.7607843137254902, 0.6470588235294118, 0.3])
mean = np.zeros(100)
x_mean = np.linspace(0, 100, 100)
for fly, fly_df in df.groupby("Fly"):
    #date, fly_number = fly.split("_")
    #beh_rois = utils.get_beh_rois("output/Fig2/behavior_prediction_results.csv", date, fly_number)
    #walking_rois = beh_rois.loc[beh_rois["Behaviour"] == "walking", "ROI"]
    walking_rois = matrix_results.loc[(matrix_results["Fly"] == fly) & (matrix_results["Variable"] == "walking"), "ROI"].unique()
    fly_df = fly_df.loc[fly_df["ROI"].isin(walking_rois), :]

    y = np.sort(fly_df.loc[fly_df["Variable"] == "vel", "rsquared"].values)[::-1]
    x = np.linspace(0, 100, len(y))
    plt.plot(x, y, color=np.copy(color))
    mean = mean + np.interp(x_mean, x, y)
mean = mean / 5
color[3] = 1
plt.plot(x_mean, mean, color=color)
plt.ylim(0, 85)
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.xlabel("% walking ROIs (neurons)")
plt.ylabel("UEV (%)")
plt.savefig("output/Fig3/panel_f.pdf")
plt.close()
