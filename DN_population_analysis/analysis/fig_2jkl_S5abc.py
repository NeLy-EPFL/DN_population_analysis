import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection
import sklearn.preprocessing
import sklearn.decomposition
import numpy as np
import utils2p

import DN_population_analysis.utils as utils


def cmap_walking(x):
    return matplotlib.colors.rgb2hex(matplotlib.cm.get_cmap("viridis")(x))

def cmap_resting(x):
    return matplotlib.colors.rgb2hex(matplotlib.cm.get_cmap("viridis_r")(x))

pc_df = pd.DataFrame()

all_trial_dirs = utils.load_exp_dirs("../../recordings.txt")

for fly_dir, trial_dirs in utils.group_by_fly(all_trial_dirs).items():
    date = utils.get_date(fly_dir)
    genotype = utils.get_genotype(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    df = utils.load_fly_data(trial_dirs, dFF=True, derivatives=True, behaviour=True)
    df = utils.reorder_ROIs(df, "output/Fig2/order_behaviour_encoding.csv")

    df = df.loc[df["Frame"] > np.min(df["Frame"]) + 100, :]
    df = df.loc[df["Frame"] < np.max(df["Frame"]) - 100, :]

    df = df.pivot(index=["Trial", "Frame", "Behaviour"], columns="ROI", values="Derivative")
    
    rois = df.columns
    
    df = df.reset_index()
        
    for t in df["Trial"].unique():
        trial_mask = df["Trial"] == t
        for beh in df["Behaviour"].unique():
            beh_mask = df["Behaviour"] == beh
            event_indices, event_number = utils2p.synchronization.event_based_frame_indices(beh_mask[trial_mask])
            df.loc[trial_mask & beh_mask, "Event indices"] = event_indices[event_indices >= 0]
            df.loc[trial_mask & beh_mask, "Event number"] = event_number[event_indices >= 0]
    df["Trial_beh_event"] = df["Trial"].map(str) + df["Behaviour"] + df["Event number"].map(str)


    PCA = sklearn.decomposition.PCA()
    scaler = sklearn.preprocessing.StandardScaler()

    beh_mask = (df["Behaviour"] == "walking").values
    valid_trial_events = df.groupby("Trial_beh_event").max().reset_index()
    valid_trial_events = valid_trial_events.loc[valid_trial_events["Event indices"] >= 50, :]
    trial_event_mask = df["Trial_beh_event"].isin(valid_trial_events["Trial_beh_event"]).values
    data = df.loc[beh_mask & trial_event_mask, rois]#.values[100:, :]
    
    PCA.fit(data)
    for i, roi in enumerate(rois):
        for pc in range(PCA.components_.shape[0]):
            row_df = pd.DataFrame({"Date": [date,], "Genotype": [genotype,], "Fly": [fly,], "ROI": [roi,], "PC": [pc,], "Loading": [PCA.components_[pc, i],]})
            pc_df = pd.concat([pc_df, row_df], axis=0)

    pcs = PCA.transform(df[rois])

    pcs = pcs / (np.max(pcs, axis=0) - np.min(pcs, axis=0))#[np.newaxis, :]

    plt.figure(figsize=(1,1))
    ax = plt.gca()
    plt.bar(np.arange(1, 7), PCA.explained_variance_ratio_[:6] * 100, color="gray")
    plt.plot(np.arange(1, 7), np.cumsum(PCA.explained_variance_ratio_[:6]) * 100, marker="o", color="black")
    plt.ylabel("variance explained (%)")
    plt.xlabel("PC")
    plt.ylim(0, 90)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks(ticks=np.arange(1, 7))
    plt.savefig(f"output/Fig2/panel_j_Fly{fly}.pdf", transparent=True)
    plt.close()

    df["Color"] = df["Behaviour"].map(utils.BEHAVIOUR_COLOURS)
    df["Color"] = df["Color"].replace({"#FFFFFF": "#949494"})

    for (t, b, e), sub_df in df.groupby(["Trial", "Behaviour", "Event number"]):
        if b == "walking":
            df.loc[(df["Trial"] == t) & (df["Behaviour"] == b) & (df["Event number"] == e), "Color"] = (sub_df["Event indices"] / sub_df["Event indices"].max()).map(cmap_walking)
        elif b == "resting":
            df.loc[(df["Trial"] == t) & (df["Behaviour"] == b) & (df["Event number"] == e), "Color"] = (sub_df["Event indices"] / sub_df["Event indices"].max()).map(cmap_resting)
    df["Alpha"] = 0.1
    
    for behaviour in ["walking", "resting"]:
        mask = df["Behaviour"] == behaviour

        for pc_x, pc_y in itertools.combinations(np.arange(2), 2):
            fig = plt.figure(figsize=(1.5, 1))
            ax = plt.gca()
            ax.set_aspect("equal")
            segments = []
            colors = []
            for (t, e), sub_df in df.loc[mask, :].groupby(["Trial", "Event number"]):
                event_mask = mask & (df["Trial"] == t) & (df["Event number"] == e)
                points = pcs[event_mask][:, [pc_x, pc_y]].reshape(-1, 1, 2)
                segments.append(np.concatenate([points[:-1], points[1:]], axis=1))
                for hex_col, alpha in zip(df["Color"].values[event_mask][:-1], df["Alpha"].values[event_mask][:-1]):
                    col = matplotlib.colors.to_rgb(hex_col)
                    colors.append((*col, alpha))
            segments = np.concatenate(segments, axis=0)
            lc = LineCollection(segments, colors=colors, linewidth=0.5)

            for i, roi in enumerate(rois):
                if PCA.components_[pc_x, i] ** 2 + PCA.components_[pc_y, i] ** 2 > 0.1:
                    plt.arrow(0, 0, PCA.components_[pc_x, i] / 2, PCA.components_[pc_y, i] / 2, zorder=10000, width=0.0005, head_width=0.005)
                    plt.text(PCA.components_[pc_x, i] * 1.1, PCA.components_[pc_y, i] * 1.1, f"ROI {roi}", zorder=10000, fontsize=4)
            plt.xlim(-0.7, 0.7)
            plt.ylim(-0.7, 0.7)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            if behaviour == "walking":
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
            elif behaviour == "resting":
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis_r"), ax=ax)

            panel_letter = "k" if behaviour == "walking" else "l"
            plt.savefig(f"output/Fig2/panel_{panel_letter}_pc{pc_x}_pc{pc_y}_{behaviour}_Fly{fly}.pdf", transparent=True)
            plt.close()

            # That data has to be plotted separately because Illustrator cannot handle number of line segments
            plt.figure(figsize=(1, 1)) 
            ax = plt.gca()
            plt.xlim(-0.7, 0.7)
            plt.ylim(-0.7, 0.7)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            line = ax.add_collection(lc)
            for i in range(0, segments.shape[0], 100):
                x = segments[i, 0, 0]
                y = segments[i, 0, 1]
                dx = segments[i, 1, 0] - x
                dy = segments[i, 1, 1] - y
                length = np.sqrt(dx ** 2 + dy ** 2)
                dx = dx / length / 500
                dy = dy / length / 500
                plt.arrow(x, y, dx, dy, color=colors[i][:3], linewidth=0.5)
            plt.savefig(f"output/Fig2/panel_{panel_letter}_pc{pc_x}_pc{pc_y}_{behaviour}_Fly{fly}.png", transparent=True, dpi=2000)

            plt.close()

pc_df.to_csv("output/Fig2/principal_components.csv")
