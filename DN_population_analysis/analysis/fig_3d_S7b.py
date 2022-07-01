import pandas as pd
import seaborn as sns
import numpy as np
import scipy.cluster
from matplotlib import pyplot as plt
import matplotlib.colors

import DN_population_analysis.utils as utils


N = 256
vals = np.ones((N, 4))
vals[:N//2, 0] = np.linspace(90/256, 1, N//2)
vals[:N//2, 1] = np.linspace(39/256, 1, N//2)
vals[:N//2, 2] = np.linspace(41/256, 1, N//2)
vals[N//2:, 0] = np.linspace(1, 102/256, N//2)
vals[N//2:, 1] = np.linspace(1, 194/256, N//2)
vals[N//2:, 2] = np.linspace(1, 165/256, N//2)
speed_colormap = matplotlib.colors.ListedColormap(vals)

def cluster_corr(corr_array, inplace=False):
    """
    Function from: https://wil.yegelwel.com/cluster-correlation-matrix/

    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    n_neurons = corr_array.shape[0]
    pairwise_distances = np.zeros(int(n_neurons * (n_neurons - 1) / 2))
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            index = n_neurons * i + j - ((i + 2) * (i + 1)) // 2
            pairwise_distances[index] = 1 - corr_array[i][j]

    linkage = scipy.cluster.hierarchy.linkage(pairwise_distances, method='ward', optimal_ordering=True)
    idx = scipy.cluster.hierarchy.leaves_list(linkage)

    # Invert order if biggest cluster is at the end
    if isinstance(corr_array, pd.DataFrame):
        weights = np.mean(corr_array.iloc[idx, :].values, axis=1)
    else:
        weights = np.mean(corr_array[idx, :].values, axis=1)
    weights = weights / np.sum(weights)
    centroid = np.sum(weights * np.arange(len(idx)))
    if centroid > len(idx) / 2:
        idx = idx[::-1]

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


directories = utils.load_exp_dirs("../../recordings.txt")

matrix_results = pd.read_csv("output/Fig2/behavior_prediction_results.csv")
matrix_results = matrix_results.loc[matrix_results["Context"] == "all", :]
matrix_results = matrix_results.groupby(["Fly", "Variable", "ROI"]).mean().reset_index()
matrix_results.loc[matrix_results["rsquared"] < 0.05, "Variable"] = "undefined"
    
beh_colors = [utils.BEHAVIOUR_COLOURS[beh] for beh, val in sorted(utils.BEHAVIOUR_LINEAR.items(), key=lambda item: item[1]) if beh[0].islower()]
beh_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("behaviour", beh_colors)

order_df = pd.DataFrame()

for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    matrix_results_fly = matrix_results.loc[matrix_results["Fly"] == f"{date}_{fly}", :]
    matrix_results_fly = matrix_results_fly.sort_values("rsquared", ascending=False)
    matrix_results_fly = matrix_results_fly.drop_duplicates("ROI", keep="first")
    matrix_results_fly = matrix_results_fly[["Variable", "ROI", "rsquared"]]
    matrix_results_fly["colour"] = matrix_results_fly["Variable"].map(utils.BEHAVIOUR_COLOURS)
    matrix_results_fly["linear"] = matrix_results_fly["Variable"].map(utils.BEHAVIOUR_LINEAR)
    hex2rgb = lambda x: tuple(int(x.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
    matrix_results_fly["colour"] = matrix_results_fly["colour"].map(hex2rgb, na_action="ignore")
    linear_mapping = {roi: value for roi, value in zip(matrix_results_fly["ROI"], matrix_results_fly["linear"])}
    
    corr_mats = []
    turning_vectors = []
    vel_vectors = []
    acv_vectors = []
    msc_vectors = []
   
    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, fictrac=True, angles=False, active=False, odor=True, joint_positions=False)

    for trial, trial_df in fly_df.groupby("Trial"):
        roi_data = trial_df.reset_index().pivot(index=("Trial", "Frame"), columns="ROI", values="dFF").sort_index()
        corr_mats.append(roi_data.corr())

        turning_coefs = trial_df.loc[trial_df["Behaviour"] == "walking", :].groupby("ROI").apply(lambda x: utils.orthogonalize(x["dFF"], x["conv_vel"]).corr(utils.orthogonalize(x["conv_turn"], x["conv_vel"])))
        turning_vectors.append(turning_coefs)
        
        vel_coefs = trial_df.loc[trial_df["Behaviour"] == "walking", :].groupby("ROI").apply(lambda x: utils.orthogonalize(x["dFF"], x["conv_turn"]).corr(utils.orthogonalize(x["conv_vel"], x["conv_turn"])))
        vel_vectors.append(vel_coefs)
        
        acv_coefs = trial_df.groupby("ROI").apply(lambda x: x["dFF"].corr(x["Odor"] == "ACV"))
        acv_vectors.append(acv_coefs)
        msc_coefs = trial_df.groupby("ROI").apply(lambda x: x["dFF"].corr(x["Odor"] == "MSC"))
        msc_vectors.append(msc_coefs)

    corr_mats = pd.concat(corr_mats)
    corr_mat = corr_mats.groupby(corr_mats.index).mean()
    
    turning_vectors = pd.concat(turning_vectors)
    turning_coefs = turning_vectors.groupby(turning_vectors.index).mean()
    
    vel_vectors = pd.concat(vel_vectors)
    vel_coefs = vel_vectors.groupby(vel_vectors.index).mean()
    
    acv_vectors = pd.concat(acv_vectors)
    acv_coefs = acv_vectors.groupby(acv_vectors.index).mean()
    msc_vectors = pd.concat(msc_vectors)
    msc_coefs = msc_vectors.groupby(msc_vectors.index).mean()

    corr_mat = cluster_corr(corr_mat)
    rois = corr_mat.columns.to_frame()
    turning_coefs = turning_coefs.reindex(rois.index)
    vel_coefs = vel_coefs.reindex(rois.index)

    fly_order_df = pd.DataFrame()
    fly_order_df["ROI"] = rois.index
    fly_order_df["Fly"] = f"{date}_{fly}"
    fly_order_df["Fly_ROI"] =  fly_order_df["Fly"].map(str) + " " + fly_order_df["ROI"].map(str)
    fly_order_df["New_ROI"] = np.arange(1, fly_order_df.shape[0] + 1)
    order_df = order_df.append(fly_order_df)

    fig, axes = plt.subplots(2, 4, gridspec_kw={'height_ratios': [1, 10], 'width_ratios': [10, 1, 1, 1]}, figsize=(1.3, 1))
    
    x = rois["ROI"].map(linear_mapping).values[np.newaxis]
    axes[0, 0].pcolor(x, cmap=beh_cmap)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Behaviour", fontsize=4, pad=2)

    max_coef = 1
    c = axes[1, 1].pcolor(turning_coefs.values[::-1, np.newaxis], cmap="bwr", vmin=-max_coef, vmax=max_coef)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Turning", rotation="vertical", fontsize=4, pad=2)
    
    c = axes[1, 2].pcolor(vel_coefs.values[::-1, np.newaxis], cmap=speed_colormap, vmin=-max_coef, vmax=max_coef)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Speed", rotation="vertical", fontsize=4, pad=2)
    
    axes[1, 3].axis("off")
    cb = plt.colorbar(c, ax=axes[1, 3])
    cb.outline.set_visible(False)

    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    sns.heatmap(corr_mat, ax=axes[1, 0], cbar=False, vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
    plt.subplots_adjust(top=0.85, right=0.85, hspace=0.05, wspace=0.05, left=0.2, bottom=0.15)
    plt.savefig(f"output/Fig3/panel_d_Fly{fly}.pdf", transparent=True)
    plt.close()

order_df.to_csv("output/Fig3/order_clustering.csv")
