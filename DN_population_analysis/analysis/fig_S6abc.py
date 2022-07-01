import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils2p.synchronization

import DN_population_analysis.utils as utils


trial_dirs = utils.load_exp_dirs("../../recordings.txt")

df = utils.load_fly_data(trial_dirs, fictrac=True, behaviour=True)
df["Forward epoch"] = np.nan
df["Forward epoch index"] = np.nan
df["Backward epoch"] = np.nan
df["Backward epoch index"] = np.nan

for (fly, trial), trial_df in df.groupby(["Fly", "Trial"]):
    beh_binary = trial_df["Behaviour"].isin(["walking", "background"])
    forward_binary = (trial_df["vel"] > 0.1) & beh_binary
    backward_binary = (trial_df["vel"] < -0.1) & beh_binary

    forward_binary = utils.hysteresis_filter(forward_binary, n=10, n_false=5)
    backward_binary = utils.hysteresis_filter(backward_binary, n=10, n_false=5)

    forward_epoch_index, forward_epoch = utils2p.synchronization.event_based_frame_indices(forward_binary)
    backward_epoch_index, backward_epoch = utils2p.synchronization.event_based_frame_indices(backward_binary)

    df.loc[(df["Fly"] == fly) & (df["Trial"] == trial), "Forward epoch"] = forward_epoch
    df.loc[(df["Fly"] == fly) & (df["Trial"] == trial), "Forward epoch index"] = forward_epoch_index
    df.loc[(df["Fly"] == fly) & (df["Trial"] == trial), "Backward epoch"] = backward_epoch
    df.loc[(df["Fly"] == fly) & (df["Trial"] == trial), "Backward epoch index"] = backward_epoch_index

assert len(np.where((df["Forward epoch index"] >= 0) & (df["Backward epoch index"] >= 0))[0]) == 0

df = df.loc[(df["Forward epoch index"] >= 0) | (df["Backward epoch index"] >= 0), :]

n_forward_epochs = df.groupby(["Fly", "Trial"])["Forward epoch"].nunique().rename("Forward")
n_backward_epochs = df.groupby(["Fly", "Trial"])["Backward epoch"].nunique().rename("Backward")
n_epochs = pd.concat([n_forward_epochs, n_backward_epochs], axis=1)
n_epochs = n_epochs.reset_index()
n_epochs = n_epochs.melt(id_vars=["Fly", "Trial"], var_name="Walking direction", value_name="# Epochs")

length_forward_epochs = df.loc[df["Forward epoch index"] >= 0, :].groupby(["Fly", "Trial", "Forward epoch"])["Forward epoch index"].max().rename("Forward")
length_backward_epochs = df.loc[df["Backward epoch index"] >= 0, :].groupby(["Fly", "Trial", "Backward epoch"])["Backward epoch index"].max().rename("Backward")
length_epochs = pd.concat([length_forward_epochs, length_backward_epochs], axis=1)
length_epochs = length_epochs.reset_index()
length_epochs = length_epochs.drop(columns="level_2")
length_epochs = length_epochs.melt(id_vars=["Fly", "Trial"], var_name="Walking direction", value_name="Epoch length")
length_epochs = length_epochs.dropna()

speed_forward_epochs = df.loc[df["Forward epoch index"] >= 0, :].groupby(["Fly", "Trial", "Forward epoch"])["vel"].quantile(q=0.95).rename("Forward")
speed_backward_epochs = df.loc[df["Backward epoch index"] >= 0, :].groupby(["Fly", "Trial", "Backward epoch"])["vel"].quantile(q=0.05).rename("Backward")
speed_epochs = pd.concat([speed_forward_epochs, speed_backward_epochs], axis=1)
speed_epochs = speed_epochs.reset_index()
speed_epochs = speed_epochs.drop(columns="level_2")
speed_epochs = speed_epochs.melt(id_vars=["Fly", "Trial"], var_name="Walking direction", value_name="Speed")
speed_epochs = speed_epochs.dropna()

sns.set(rc=utils.SEABORN_RC)
#sns.set_theme(style="ticks")

x = "Walking direction"
n = 700
order = ["Forward", "Backward"]

plt.figure(figsize=(2,2))
y = "# Epochs"
sns.swarmplot(data=n_epochs, x=x, y=y, hue="Fly", size=2, order=order)
sns.boxplot(data=n_epochs, x=x, y=y, showfliers=False, order=order)
plt.legend([], [], frameon=False)
plt.savefig("output/FigS6/panel_a.pdf", transparent=True)
plt.close()

length_epochs["Epoch length (s)"] = length_epochs["Epoch length"] / 100
subsampled_length_epochs = length_epochs.groupby(x).sample(n=n, replace=False, random_state=42)
plt.figure(figsize=(2,2))
y = "Epoch length (s)"
sns.swarmplot(data=subsampled_length_epochs, x=x, y=y, hue="Fly", size=0.75, order=order)
sns.boxplot(data=length_epochs, x=x, y=y, showfliers=False, order=order)
plt.legend([], [], frameon=False)
plt.savefig("output/FigS6/panel_b.pdf", transparent=True)
plt.close()

subsampled_speed_epochs = speed_epochs.groupby(x).sample(n=n, replace=False, random_state=42)
plt.figure(figsize=(2,2))
y = "Speed"
sns.swarmplot(data=subsampled_speed_epochs, x=x, y=y, hue="Fly", size=0.75, order=order)
sns.boxplot(data=speed_epochs, x=x, y=y, showfliers=False, order=order)
plt.legend([], [], frameon=False)
plt.savefig("output/FigS6/panel_c.pdf", transparent=True)
plt.close()


