import itertools

import sklearn.discriminant_analysis
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import imblearn.over_sampling
import seaborn as sns
import statannotations.Annotator

import DN_population_analysis.utils as utils


results_df = pd.DataFrame()

all_trial_dirs = utils.load_exp_dirs("../../recordings.txt")

for fly_dir, trial_dirs in utils.group_by_fly(all_trial_dirs).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)
    fly_df = utils.load_fly_data(trial_dirs, odor=True, behaviour=True, dFF=True, fictrac=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["ACV", "MSC", "H2O"]), :]

    conv_behaviours = []
    t = np.arange(30000) / 100
    for beh in fly_df["Behaviour"].unique():
        conv_beh = f"conv_{beh}"
        fly_df[conv_beh] = utils.convolve_with_crf(t, fly_df["Behaviour"] == beh, fly_df["Trial"])
        conv_behaviours.append(conv_beh)

    regressors = conv_behaviours + ["conv_turn", "conv_vel", "conv_side"]
    bounds = 3 * np.array([[np.NINF, np.inf],] + len(conv_beh) * [[0, np.inf],]).transpose()
    fly_df = utils.get_residuals(fly_df, regressors, bounds)

    for behaviour in ("all",):
        if behaviour == "all":
            beh_df = fly_df
        else:
            beh_df = fly_df.loc[fly_df["Behaviour"] == behaviour, :]

        for trial, trial_df in beh_df.groupby("Trial"):
            for odor1, odor2 in [("H2O", "ACV"), ("H2O", "MSC")]:
                for data_type in ("neural_residuals", "conv_beh"):
                    if odor1 != "all" and odor2 != "all":
                        sub_df = trial_df.loc[trial_df["Odor"].isin([odor1, odor2]), :]
                    else:
                        sub_df = trial_df.copy()
                        sub_df.loc[sub_df["Odor"].isin(["ACV", "MSC"]), "Odor"] = "all"
                    
                    if data_type == "neural":
                        sub_df = sub_df.pivot(index=["Frame", "Odor"], columns="ROI", values="dFF")
                    elif data_type == "neural_residuals":
                        sub_df = sub_df.pivot(index=["Frame", "Odor"], columns="ROI", values="residual_dFF")
                    elif data_type == "conv_beh":
                        sub_df = sub_df.drop(columns=["ROI", "dFF"])
                        sub_df = sub_df.drop_duplicates()
                        sub_df = sub_df.set_index(["Frame", "Odor"])
                        sub_df = sub_df[["conv_vel", "conv_side", "conv_turn"] + conv_behaviours]
                    elif data_type == "ball":
                        sub_df = sub_df.drop(columns=["ROI", "dFF"])
                        sub_df = sub_df.drop_duplicates()
                        sub_df = sub_df.set_index(["Frame", "Odor"])
                        sub_df = sub_df[["vel", "side", "turn"]]
                    else:
                        raise ValueError("Unknown data_type")

                    columns = sub_df.columns

                    X = sub_df.values
                    sub_df = sub_df.reset_index()
                    label_encoder = sklearn.preprocessing.LabelEncoder()
                    y = label_encoder.fit_transform(sub_df["Odor"])

                    if len(np.unique(y)) == 1:
                        print("continuing because there is only one stimulus type")
                        continue

                    scaler = sklearn.preprocessing.StandardScaler()
                    X = scaler.fit_transform(X)

                    oversampler = imblearn.over_sampling.SMOTE()
                    X, y = oversampler.fit_resample(X, y)

                    skf = sklearn.model_selection.StratifiedKFold(n_splits=5)
                    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
                        lda.fit(X_train, y_train)
                        score = lda.score(X_test, y_test)
                        weights = lda.coef_

                        transformed = np.squeeze(lda.transform(X))

                        for i, column in enumerate(columns):
                            
                            if len(np.unique(X[:, i])) == 1 or len(np.unique(transformed)) == 1:
                                continue
                            corr, p_val = scipy.stats.pearsonr(X[:, i], transformed)

                            row_df = pd.DataFrame({
                                        "Fly":      [f"{date}_{fly}",],
                                        "Trial":    [trial,],
                                        "Fold":     [fold,],
                                        "Odor1":    [odor1,],
                                        "Odor2":    [odor2,],
                                        "Score":    [score,],
                                        "ROI":      [column,],
                                        "Pearson":  [corr,],
                                        "P-value":  [p_val,],
                                        "Type":     [data_type,],
                                        "Behaviour":[behaviour,],
                                        "weight":   [weights[0, i]],
                                        })
                            results_df = results_df.append(row_df)

    results_df.to_csv("output/Fig4/lda_results.csv")

df = pd.read_csv("output/Fig4/lda_results_old.csv")

df = df.groupby(["Fly", "Trial", "Odor1", "Odor2", "ROI", "Type", "Behaviour"]).mean().reset_index()
df["Combination"] = df["Odor1"] + " " + df["Odor2"]
df["abs_pearson"] = np.abs(df["Pearson"])

plt.figure(figsize=(1, 1.5))
sns.set_theme(style="ticks", rc=utils.SEABORN_RC)

box_df = df.drop_duplicates(subset=["Fly", "Combination", "Trial", "Score", "Type", "Behaviour"])

box_df = box_df.loc[box_df["Type"].isin(["neural_residuals", "conv_beh"]), :]
box_df = box_df.loc[box_df["Odor2"] != "all", :]
box_df = box_df.loc[box_df["Odor1"] == "H2O", :]
box_df = box_df.loc[box_df["Behaviour"] == "all", :]
order = ["ACV\nneural_residuals", "ACV\nconv_beh", "MSC\nneural_residuals", "MSC\nconv_beh"]

box_df["Combination"] = box_df["Odor2"] + "\n" + box_df["Type"]
box_df = box_df.sort_values("Combination")
x = "Combination"
y = "Score"
hue = "Fly"
ax = sns.boxplot(x=x, y=y, data=box_df, order=order)
ax = sns.swarmplot(x=x, y=y, data=box_df, hue=hue, size=2, order=order)
ax.set_ylim(0.45, 1.05)
pairs=[("ACV\nneural_residuals", "ACV\nconv_beh"),
       ("MSC\nneural_residuals", "MSC\nconv_beh"),
      ]
annotator = statannotations.Annotator.Annotator(ax, pairs, data=box_df, x=x, y=y, order=order)
annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", line_width=0.5)
annotator.apply_and_annotate()
plt.axhline(y=0.5, linestyle="--", color="gray")
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("output/Fig4/panel_a.pdf", transparent=True)
plt.close()
quit()

neural_df = df.loc[df["Type"] == "neural", :]
neural_df = neural_df.groupby(["Behaviour", "Fly", "ROI", "Combination"]).mean().reset_index()
for combination, comb_neural_df in neural_df.groupby("Combination"):
    fig = plt.figure()
    for beh, beh_comb_neural_df in comb_neural_df.groupby("Behaviour"):
        x_summary = np.linspace(0, 1, 100)
        y_summary = []
        for fly, sub_df in beh_comb_neural_df.groupby("Fly"):
            y = sub_df["abs_pearson"].values
            y = np.sort(y)[::-1]
            x = np.linspace(0, 1, len(y))
            plt.plot(x, y, color=utils.behaviour_colours[beh], alpha=0.1) 
            y_summary.append(np.interp(x_summary, x, y))
        y_summary = np.mean(y_summary, axis=0)
        plt.plot(x_summary, y_summary, color=utils.behaviour_colours[beh], label=utils.rename_behaviour(beh))
    plt.legend()
    plt.savefig(f"otuput/Fig4/pearson_{combination}.pdf")
    plt.close()

corr_df = pd.DataFrame()
for (combination, fly), sub_df in neural_df.groupby(["Combination", "Fly"]):
    spearman = sub_df.pivot(columns="Behaviour", index="ROI", values="Pearson").corr(method="spearman")
    row_df = pd.DataFrame({"Fly": [fly,],
                           "Combination": [combination,],
                           "Spearman rank correlation": [spearman.values[0, 1],],
                         })
    corr_df = corr_df.append(row_df)

ax = sns.boxplot(x="Combination", y="Spearman rank correlation", data=corr_df)
ax = sns.swarmplot(x="Combination", y="Spearman rank correlation", data=corr_df, hue="Fly", size=2)
#ax.set_ylim(0.45, 1.05)
#plt.axhline(y=0.5, linestyle="--", color="gray")
plt.legend([], [], frameon=False)
plt.setp(ax.artists, facecolor="w")
plt.savefig("FigContext/spearman.pdf", transparent=True)
plt.close()

df = df.loc[df["Type"] == "neural_residuals", :]
df = df.groupby(["Fly", "Combination", "ROI"]).mean().reset_index()

for (combination, fly), combination_df in df.groupby(["Combination", "Fly"]):
    date, fly_number = fly.split("_")
    trial_dir = utils.get_trial_dir(date, "Ci1xG23", fly_number, 1).rstrip("/") + "_001_ref/"
    print(trial_dir)
    output_file = f"FigContext/lda_corr_{date}_{combination.replace(' ', '_')}.pdf"
    utils.roi_location_plot(trial_dir,
                            combination_df["ROI"].values.astype(int),
                            combination_df["Pearson"].values,
                            combination_df["abs_pearson"].values,
                            output_file,
                            vmax_radius=1,
                            vmin_colour=-1,
                            vmax_colour=1,
                            cmap=matplotlib.cm.bwr,
                            numbers=False,
                            figsize=(1.5, 3),
                            percent=False,
                           )
