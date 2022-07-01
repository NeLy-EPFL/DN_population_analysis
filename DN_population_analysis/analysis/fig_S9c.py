import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from statsmodels.stats.multitest import multipletests
import statannotations.Annotator
import DN_population_analysis.utils as utils


df = pd.read_csv("output/Fig2/behavior_prediction_results.csv", index_col=0).reset_index(drop=True)
df = df.loc[df["Context"] != "all", :]
df = df.loc[df["Variable"].isin(["walking", "head_grooming"]), :]
df = df.pivot(index=["Fly", "ROI", "Fold", "Trial", "Variable", "Subset"], columns="Context", values="rsquared").reset_index()

p_value_df_acv = df.groupby(["Fly", "ROI", "Variable"]).apply(lambda t: scipy.stats.mannwhitneyu(t["H2O"].dropna(), t["ACV"].dropna())[1]).to_frame(name="p")
p_value_df_acv["Odor"] = "ACV"
p_value_df_msc = df.groupby(["Fly", "ROI", "Variable"]).apply(lambda t: scipy.stats.mannwhitneyu(t["H2O"].dropna(), t["MSC"].dropna())[1]).to_frame(name="p")
p_value_df_msc["Odor"] = "MSC"

p_value_df = pd.concat([p_value_df_acv, p_value_df_msc], axis=0)
p_value_df["p"].shape
p_value_df["p corr"] = multipletests(p_value_df["p"], alpha=0.05, method="bonferroni")[1]

p_value_df["ratio"] = p_value_df["p corr"] /  p_value_df["p"]

print(p_value_df.sort_values("p corr"))

significant = p_value_df.loc[p_value_df["p corr"] < 0.05, :].reset_index()

for (fly, roi, var, odor), _ in significant.groupby(["Fly", "ROI", "Variable", "Odor"]):
    sub_df = df.loc[(df["Fly"] == fly) &
                    (df["ROI"] == roi) &
                    (df["Variable"] == var), :]

    sub_df = pd.melt(sub_df, id_vars=["Fold", "Trial", "Subset"], value_vars=["H2O", "ACV", "MSC"], var_name="Context", value_name="rsquared")
    sub_df["rsquared"] = sub_df["rsquared"] * 100

    plt.figure(figsize=(1.6, 1.3))
    ax = sns.stripplot(x="Context", y="rsquared", data=sub_df, hue="Trial")
    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Context",
            y="rsquared",
            data=sub_df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
    pairs=[("H2O", "ACV"),
           ("H2O", "MSC"),
          ]
    annotator = statannotations.Annotator.Annotator(ax, pairs, data=sub_df, x="Context", y="rsquared")
    annotator.configure(test="Mann-Whitney", text_format="star", loc="outside", line_width=0.5)
    annotator.apply_and_annotate()
    plt.ylim(-5, 100)
    plt.ylabel("R^2")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.get_legend().remove()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"output/FigS9/panel_c_{fly}_ROI{int(roi)}_{var}.pdf", transparent=True)
    plt.close()

