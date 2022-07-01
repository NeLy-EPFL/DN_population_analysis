import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import DN_population_analysis.utils as utils


masked_df = pd.read_csv("output/Fig3/ball_rot_prediction_population_masked_results.csv")
masked_df = masked_df.groupby(["Variable", "Fly", "Trial"]).mean().reset_index()

masked_df = masked_df.loc[masked_df["Variable"].isin(["conv_turn", "conv_vel"]), :]
masked_df.loc[masked_df["Variable"] == "turn", "Variable"] = "Angular velocity"
masked_df.loc[masked_df["Variable"] == "vel", "Variable"] = "Speed"
masked_df["rsquared"] = masked_df["rsquared"] * 100
masked_df = masked_df.rename(columns={"rsquared": r'%$R^2$'})

sns.set(rc=utils.SEABORN_RC)
sns.set_theme(style="ticks")

ax = sns.boxplot(x="Variable", y=r'%$R^2$', data=masked_df)
ax = sns.swarmplot(x="Variable", y=r'%$R^2$', data=masked_df, hue="Fly", size=2)
ax.set_ylim((0, 90))
plt.tight_layout()
plt.savefig("output/Fig3/panel_b.pdf")
plt.close()
