import numpy as np
import pandas as pd
import seaborn as sns
import statannotations.Annotator
from matplotlib import pyplot as plt
import utils2p

import DN_population_analysis.utils as utils

directories = utils.load_exp_dirs("../../recordings.txt")


def hex2rgb(x):
    return [int(x.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [
        1,
    ]


beh_prob_df = pd.DataFrame()
walking_df = pd.DataFrame()

for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    genotype = utils.get_genotype(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs,
                                 dFF=False,
                                 behaviour=True,
                                 fictrac=True,
                                 angles=False,
                                 active=False,
                                 odor=True,
                                 joint_positions=False)
    fly_df = fly_df.loc[fly_df["Odor"].isin([
        "H2O",
        "ACV",
        "MSC",
    ]), :]

    behaviours = fly_df["Behaviour"].unique()
    behaviours = behaviours[behaviours != "background"]
    odors = fly_df["Odor"].unique()

    for odor in odors:
        for trial, trial_df in fly_df.groupby("Trial"):
            binary_seq = trial_df["Odor"] == odor
            event_based_frame_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(
                binary_seq)
            trial_df["event_number"] = event_numbers
            trial_df = trial_df.loc[trial_df["Odor"] == odor, :]
            for event, event_df in trial_df.groupby("event_number"):
                for beh in behaviours:
                    prob = np.sum(
                        event_df["Behaviour"] == beh) / event_df.shape[0]
                    row_df = pd.DataFrame({
                        "Date": (date, ),
                        "Trial": (trial, ),
                        "Odor": (odor, ),
                        "Behaviour": (utils.rename_behaviour(beh), ),
                        "Probability": (prob, ),
                    })
                    beh_prob_df = beh_prob_df.append(row_df)

            binary_seq = trial_df["Behaviour"] == "walking"
            event_based_frame_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(
                binary_seq)
            trial_df["event_number"] = event_numbers
            trial_df = trial_df.loc[trial_df["Odor"] == odor, :]
            for event, event_df in trial_df.groupby("event_number"):
                mean_vel = event_df["vel"].mean()
                mean_turn = event_df["turn"].mean()
                row_df = pd.DataFrame({
                    "Date": (date, ),
                    "Trial": (trial, ),
                    "Odor": (odor, ),
                    "Speed (mm/s)": (mean_vel, ),
                    "Turning (°/s)": (mean_turn, ),
                })
                walking_df = walking_df.append(row_df)

sns.set_theme(style="ticks")
sns.set(rc=utils.SEABORN_RC)

plt.figure(figsize=(1, 1.5))
behaviours = [
    "Resting", "Walking", "Posterior movements", "Head grooming",
    "Front leg rubbing"
]
order = [
    beh + "\n" + odor for beh in behaviours
    for odor in ["H2O", "ACV", "MSC", "Z"]
]
palette = [
    utils.BEHAVIOUR_COLOURS["H2O"],
    utils.BEHAVIOUR_COLOURS["ACV"],
    utils.BEHAVIOUR_COLOURS["MSC"],
]
beh_prob_df["Behavior+Odor"] = beh_prob_df["Behaviour"] + \
    "\n" + beh_prob_df["Odor"]
# hack to create space between groups
empty_categories = [
    "Resting\nZ", "Walking\nZ", "Posterior movements\nZ", "Head grooming\nZ",
    "Front leg rubbing\nZ"
]
for category in empty_categories:
    row_df = pd.DataFrame({
        "Behavior+Odor": [
            category,
        ],
        "Probability": [
            0,
        ],
        "Date": [
            210830,
        ]
    })
    beh_prob_df = beh_prob_df.append(row_df)

x = "Behavior+Odor"
y = "Probability"
plt.figure(figsize=(2.2, 1.5))
ax = sns.boxplot(x=x, y=y, data=beh_prob_df, order=order, showfliers=False)
ax = sns.swarmplot(x=x,
                   y=y,
                   hue="Date",
                   data=beh_prob_df,
                   order=order,
                   size=0.5)
ax.set_ylim(0, 1)
pairs = [(f"{beh}\n{odor1}", f"{beh}\n{odor2}")
         for odor1, odor2 in [("H2O", "ACV"), ("ACV", "MSC"), ("H2O", "MSC")]
         for beh in behaviours]
annotator = statannotations.Annotator.Annotator(ax,
                                                pairs,
                                                data=beh_prob_df,
                                                x=x,
                                                y=y,
                                                order=order)
annotator.configure(test="Mann-Whitney",
                    text_format="star",
                    loc="inside",
                    line_width=0.5)
annotator.apply_and_annotate()
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("output/FigS8/panel_a.pdf", transparent=True)
plt.close()

n = walking_df["Odor"].value_counts().min()
subsampled_walking_df = walking_df.groupby("Odor").sample(n=n,
                                                          replace=False,
                                                          random_state=42)

x = "Odor"
y = "Speed (mm/s)"
ax = sns.boxplot(x=x,
                 y=y,
                 data=walking_df,
                 palette=palette,
                 order=["H2O", "ACV", "MSC"],
                 showfliers=False)
ax = sns.swarmplot(x=x,
                   y=y,
                   data=subsampled_walking_df,
                   hue="Date",
                   size=0.5,
                   order=["H2O", "ACV", "MSC"])
pairs = [("H2O", "ACV"), ("ACV", "MSC"), ("H2O", "MSC")]
annotator = statannotations.Annotator.Annotator(ax,
                                                pairs,
                                                data=walking_df,
                                                x=x,
                                                y=y)
annotator.configure(test="Mann-Whitney",
                    text_format="star",
                    loc="outside",
                    line_width=0.5)
annotator.apply_and_annotate()
ax.set_ylim(subsampled_walking_df[y].min() - 0.3,
            subsampled_walking_df[y].max() + 0.3)
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("output/FigS8/panel_b.pdf", transparent=True)
plt.close()

x = "Odor"
y = "Turning (°/s)"
order = ["H2O", "ACV", "MSC"]
ax = sns.boxplot(x=x,
                 y=y,
                 data=walking_df,
                 order=order,
                 showfliers=False,
                 palette=palette)
ax = sns.swarmplot(x=x,
                   y=y,
                   data=subsampled_walking_df,
                   hue="Date",
                   size=0.5,
                   order=order)
pairs = [("H2O", "ACV"), ("ACV", "MSC"), ("H2O", "MSC")]
annotator = statannotations.Annotator.Annotator(ax,
                                                pairs,
                                                data=walking_df,
                                                x=x,
                                                y=y)
annotator.configure(test="Mann-Whitney",
                    text_format="star",
                    loc="outside",
                    line_width=0.5)
annotator.apply_and_annotate()
ax.set_ylim(subsampled_walking_df[y].min() - 10,
            subsampled_walking_df[y].max() + 10)
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("output/FigS8/panel_c.pdf", transparent=True)
plt.close()

x = "Odor"
y = "Turning (°/s)"
order = ["H2O", "ACV", "MSC"]
walking_df[y] = np.abs(walking_df[y])
subsampled_walking_df[y] = np.abs(subsampled_walking_df[y])
ax = sns.boxplot(x=x,
                 y=y,
                 data=walking_df,
                 order=order,
                 showfliers=False,
                 palette=palette)
ax = sns.swarmplot(x=x,
                   y=y,
                   data=subsampled_walking_df,
                   hue="Date",
                   size=0.5,
                   order=order)
pairs = [("H2O", "ACV"), ("ACV", "MSC"), ("H2O", "MSC")]
annotator = statannotations.Annotator.Annotator(ax,
                                                pairs,
                                                data=walking_df,
                                                x=x,
                                                y=y)
annotator.configure(test="Mann-Whitney",
                    text_format="star",
                    loc="outside",
                    line_width=0.5)
annotator.apply_and_annotate()
ax.set_ylim(subsampled_walking_df[y].min() - 10,
            subsampled_walking_df[y].max() + 10)
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("output/FigS8/panel_d.pdf", transparent=True)
plt.close()
