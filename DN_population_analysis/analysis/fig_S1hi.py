import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils2p

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")

beh_freq_df = pd.DataFrame()
beh_trans_df = pd.DataFrame()

behaviours = ["walking", "resting", "head_grooming", "foreleg_grooming", "hind_grooming"]

for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    genotype = utils.get_genotype(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=False, behaviour=True, fictrac=False, angles=False, active=False, odor=True, joint_positions=False)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]
    
    for trial, trial_df in fly_df.groupby("Trial"):
        for beh in behaviours:
            binary_seq = (trial_df["Behaviour"] == beh)

            freq = np.sum(binary_seq) / len(binary_seq)

            row_df = pd.DataFrame({"Date": (date,),
                                   "Trial": (trial,),
                                   "Behaviour": (utils.rename_behaviour(beh),),
                                   "Probability": (freq,),
                                  })
            beh_freq_df = beh_freq_df.append(row_df)

            event_based_frame_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(binary_seq)
            trial_df["event_number"] = event_numbers
            trial_df["event_frame_index"] = event_based_frame_indices
            for event, event_df in trial_df.groupby("event_number"):
                if event_df["event_frame_index"].min() > -1:
                    continue
                pre_beh = event_df.loc[event_df["event_frame_index"] == -1, "Behaviour"].values[0]
                if pre_beh == "background":
                    continue
                row_df = pd.DataFrame({"Date": (date,),
                                       "Trial": (trial,),
                                       "Behaviour": (utils.rename_behaviour(beh),),
                                       "Previous": (utils.rename_behaviour(pre_beh),),
                                      })
                beh_trans_df = beh_trans_df.append(row_df)

sns.set(utils.SEABORN_RC)
plt.figure(figsize=(2, 3.5))
order = ["Resting", "Walking", "Posterior movements", "Head grooming", "Front leg rubbing"]
palette = [utils.BEHAVIOUR_COLOURS[beh] for beh in order]
ax = sns.boxplot(x="Behaviour", y="Probability", data=beh_freq_df, order=order, palette=palette)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_color(utils.BEHAVIOUR_COLOURS[ticklabel._text])
plt.savefig("output/FigS1/panel_h.pdf", transparent=True)
plt.close()

sns.set(utils.SEABORN_RC)
plt.figure(figsize=(3, 3.5))
beh_trans_df = beh_trans_df.groupby(["Previous", "Behaviour"]).size().reset_index(name="counts")
beh_trans_df = beh_trans_df.pivot(index="Previous", columns="Behaviour", values="counts")
beh_trans_df = beh_trans_df.div(beh_trans_df.sum(axis=1), axis=0)
beh_trans_df.index = pd.CategoricalIndex(beh_trans_df.index, categories=order)
beh_trans_df.sort_index(level=0, inplace=True)
beh_trans_df = beh_trans_df[order]
ax = sns.heatmap(beh_trans_df, cmap=plt.cm.Blues, cbar_kws={"orientation": "horizontal"})
ax.xaxis.set_label_position("top")
ax.xaxis.set_ticks_position("top")
ax.set_yticklabels(order, fontsize=4, rotation=60, va="center", fontname="Arial")
for ticklabel in ax.get_yticklabels():
    ticklabel.set_color(utils.BEHAVIOUR_COLOURS[ticklabel._text])
ax.set_xticklabels(order, fontsize=4, rotation=30, ha="center", fontname="Arial")
for ticklabel in ax.get_xticklabels():
    ticklabel.set_color(utils.BEHAVIOUR_COLOURS[ticklabel._text])
ax.tick_params(axis=u"both", which=u"both", length=0)
ax.set_ylabel("Pre")
ax.set_xlabel("Post")
plt.savefig("output/FigS1/panel_i.pdf", transparent=True)
plt.close()
