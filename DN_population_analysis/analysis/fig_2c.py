import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import DN_population_analysis.utils as utils


df = pd.read_csv("output/Fig2/behavior_prediction_results.csv", index_col=0).reset_index(drop=True)
df = df.loc[df["Context"] == "all", :]
df = df.rename(columns={"Variable": "Behaviour"})
df["Behaviour"] = df["Behaviour"].apply(utils.rename_behaviour)

df = df.groupby(["Fly", "ROI", "Behaviour"]).mean().reset_index()
n_all_rois = df.groupby(["Fly", "Behaviour"]).size()
beh_df = df.sort_values("rsquared", ascending=False).drop_duplicates(["Fly", "ROI"], keep="first")
beh_df = beh_df.loc[df["rsquared"] > 0.05, :]
n_beh_rois = beh_df.groupby(["Fly", "Behaviour"]).size()
n_rois_relative = n_beh_rois / n_all_rois * 100
n_rois_relative = n_rois_relative.fillna(value=0)
n_rois_relative = n_rois_relative.to_frame(name="Fraction").reset_index()

sns.set_theme(style="ticks", rc=utils.SEABORN_RC)
order = ["Resting", "Walking", "Posterior grooming", "Head grooming", "Front leg rubbing"]
palette = [utils.BEHAVIOUR_COLOURS[beh] for beh in order]

plt.figure(figsize=(1.25, 1.75))
ax = sns.boxplot(x="Behaviour", y="Fraction", data=n_rois_relative, order=order, palette=palette)
ax = sns.swarmplot(x="Behaviour", y="Fraction", data=n_rois_relative, color=".25", size=3, order=order)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_color(utils.BEHAVIOUR_COLOURS[ticklabel._text])
plt.savefig("output/Fig2/panel_c.pdf")
