import os.path
import pickle

import pandas as pd
import numpy as np

# Only use trials that have hand labels
# Be sure to keep the same order for trial_dirs and hand_label_files

trial_dirs = [
    "/mnt/data2/FA/210830_Ci1xG23/Fly1/003_coronal",
    "/mnt/data2/FA/210910_Ci1xG23/Fly2/003_coronal",
    "/mnt/data2/FA/211026_Ci1xG23/Fly3/003_coronal",
    "/mnt/data2/FA/211027_Ci1xG23/Fly2/003_coronal",
    "/mnt/data2/FA/211029_Ci1xG23/Fly1/003_coronal",
]

hand_label_files = [
    "210830_Fly1_003_labels.csv",
    "210910_Fly2_003_labels.csv",
    "211026_Fly3_003_labels.csv",
    "211027_Fly2_003_labels.csv",
    "211029_Fly1_003_labels.csv",
]

n_frames = 540001
for trial_dir, hand_label_file in zip(trial_dirs, hand_label_files):
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")
    
    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    df = pd.read_pickle(input_file)
    
    data = df.filter(like="Angle").values.astype("float32")
    data = data[:n_frames + 1]
    
    output_file = f"markers/{descriptor}_labeled.npy"
    np.save(output_file, data)
    
    hand_labels = pd.read_csv(hand_label_file, index_col=0)
    hand_labels = hand_labels.loc[:n_frames, :]
    hand_labels["head_grooming"] = (hand_labels["eye_grooming"] + hand_labels["antennal_grooming"]).clip(upper=1)
    hand_labels["hind_grooming"] = (hand_labels["abdominal_grooming"] + hand_labels["hindleg_grooming"]).clip(upper=1)
    hand_labels = hand_labels.drop(labels=["eye_grooming", "antennal_grooming", "backward_walking", "abdominal_grooming", "hindleg_grooming"], axis=1)
    background = hand_labels.sum(axis=1) < 1
    hand_labels.loc[background, :] = 0
    hand_labels.loc[background, "background"] = 1
    hand_labels = hand_labels[["background", "resting", "walking", "head_grooming", "foreleg_grooming", "hind_grooming"]]
    hand_labels.to_csv(f"labels-hand/{descriptor}_labels.csv")
