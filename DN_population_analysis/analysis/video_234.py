import os.path
import collections
import math

import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt

import utils2p
import utils2p.synchronization
import utils_video
import utils_video.generators
import utils_ballrot

import DN_population_analysis.utils as utils


majority_func = lambda x: collections.Counter(x).most_common(1)[0][0]

source = "denoised"

beh_examples = {
        "head_grooming":    {"video_file": "/mnt/data2/FA/210830_Ci1xG23/Fly1/001_coronal/behData/images/camera_5.mp4",
                             "start": 41941},
        "walking":          {"video_file": "/mnt/data2/FA/211029_Ci1xG23/Fly1/005_coronal/behData/images/camera_5.mp4",
                             "start": 11321},
        "hind_grooming":    {"video_file": "/mnt/data2/FA/211026_Ci1xG23/Fly3/002_coronal/behData/images/camera_5.mp4",
                             "start": 18193},
        "foreleg_grooming": {"video_file": "/mnt/data2/FA/210830_Ci1xG23/Fly1/002_coronal/behData/images/camera_5.mp4",
                             "start": 18706},
        "resting":          {"video_file": "/mnt/data2/FA/210830_Ci1xG23/Fly1/001_coronal/behData/images/camera_5.mp4",
                             "start": 33384},
        }

frame_indices = {"resting": {210830: 72, 210910: 44, 211026: 70, 211027: 42, 211029: 79},
                 "walking": {210830: 69, 210910: 79, 211026: 79, 211027: 79, 211029: 79},
                 "head_grooming": {210830: 61, 210910: 34, 211026: 39, 211027: 30, 211029: 65},
                 "foreleg_grooming": {210830: 24, 210910: 29, 211026: 29, 211027: 24, 211029: 31},
                 "hind_grooming": {210830: 42, 210910: 24, 211026: 25, 211027: 32, 211029: 34},
                }

dirs = utils.load_exp_dirs("../trials_for_paper_overall_ref.txt")

for behaviour in ("head_grooming", "walking", "resting"):

    print(behaviour)

    epoch_info = pd.DataFrame()

    out_path = f"output/Videos/{behaviour}.mp4"
    generators = []
    n_valid_frames = []
    for fly_dir, trial_dirs in utils.group_by_fly(dirs).items():
        print(fly_dir)
        date = utils.get_date(fly_dir)
        fly_event_stacks = []
        mean = np.ones((500, 256, 640))
        n_events_per_fly = np.ones((500, 256, 640))
        n_r_per_fly = np.ones((500, 256, 640))
        n_l_per_fly = np.ones((500, 256, 640))
        for trial_dir in trial_dirs:
            trial = utils.get_trial_number(trial_dir)
            trial_df = utils.load_fly_data([trial_dir,], behaviour=True, fictrac=True, frames_2p=True, odor=True)
            
            binary_seq = trial_df["Behaviour"].values == behaviour
            
            
            event_based_frame_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(binary_seq)
            trial_df["Binary event"] = binary_seq
            trial_df["event_based_frame_index"] = event_based_frame_indices
            trial_df["event_number"] = event_numbers
            print("Number of events:", np.max(event_numbers) + 1)
            
            if source == "warped":
                stack = utils2p.load_img(os.path.join(trial_dir, "2p/warped_green.tif"))
                stack = fix_image_size(stack, already_cropped=False, trial_dir=trial_dir)
            elif source == "denoised":
                stack = utils2p.load_img(os.path.join(trial_dir, "2p/denoised_green.tif"))
                stack = utils.fix_image_size(stack, already_cropped=True)
            stack = stack[trial_df["Frame 2p"].min():trial_df["Frame 2p"].max() + 1]
            binary_seqs = []

            for event_number, event_df in trial_df.groupby("event_number"):
                if event_number < 0:
                    continue
                event_df = event_df.loc[(event_df["event_based_frame_index"] >= -100) &
                                        (event_df["event_based_frame_index"] < 400),
                                        :]
                epoch_info = epoch_info.append(pd.DataFrame({
                    "Date": [date,], 
                    "Trial": [trial,],
                    "Behaviour": [behaviour,],
                    "Length": [event_df.shape[0],],
                    "Start": [event_df["Frame"].min(),],
                    "Stop": [event_df["Frame"].max(),],
                }))
                if event_df["event_based_frame_index"].max() < 50 or event_df["event_based_frame_index"].min() > -100:
                    continue
                if behaviour != "noair" and "" in event_df["Odor"]:
                    continue
                if behaviour == "resting" and np.any(event_df["Odor"][:200] != "H20"):
                    continue
                
                event_indices_2p = event_df["Frame 2p"].values

                sub_stack = stack[event_indices_2p].copy()
                sub_binary_seq = event_df["Binary event"].values.copy()
                if sub_stack.shape[0] < 500:
                    extension_shape = (500 - sub_stack.shape[0], sub_stack.shape[1], sub_stack.shape[2])
                    sub_stack = np.concatenate(
                                    (sub_stack,
                                     np.full(extension_shape, np.nan)),
                                axis=0)
                    sub_binary_seq = np.concatenate((sub_binary_seq, np.full((extension_shape[0],), False)), axis=0)
                baseline_img = np.mean(sub_stack[:5], axis=0)
                dFF = (np.divide((sub_stack - baseline_img),
                                  baseline_img,
                                  out=np.zeros_like(sub_stack, dtype=np.double),
                                  where=~np.isclose(baseline_img, 0, atol=1e-10),
                                 )
                                 * 100
                       )

                mean = np.nansum(np.array([mean, dFF]), axis=0)
                n_events_per_fly = n_events_per_fly + ~np.isnan(dFF)
                binary_seqs.append(sub_binary_seq)

        mean = np.nan_to_num(mean, nan=0)
        mean = mean / n_events_per_fly
        
        mean = np.clip(mean, 0, None)
        vmin, vmax = None, None
        valid_frames = np.unique(np.where(n_events_per_fly > 7)[0])
        plt.figure()
        x = np.linspace(-1, 4, len(n_events_per_fly))
        plt.plot(x, n_events_per_fly[:, 0, 0])
        plt.axvline(x=0, color="gray", linestyle="--", label="behavior onset")
        plt.axvline(x=frame_indices[behaviour][date]/16 - 1, color="red", linestyle="--", alpha=0.5, label="frame for figure")
        plt.ylabel("# events")
        plt.xlabel("Time (s)")
        plt.ylim(0, 300)
        plt.legend()
        plt.savefig(f"{behaviour}_{date}_n_events.pdf")
        plt.close()

        utils2p.save_img(f"{date}_{behaviour}.tif", mean)

        mean = mean[valid_frames]
        n_valid_frames.append(mean.shape[0])

        generator = utils_video.generators.dff(mean, font_size=20, vmin=vmin, vmax=vmax, cmap=utils.cmap)

        dot_binary = np.sum(np.array(binary_seqs, dtype=int), axis=0)
        dot_binary = dot_binary.astype(bool)
        color = np.zeros((len(dot_binary), 3))
        color[dot_binary, 0] = 255
        color[mean.shape[0] - 1, :] = np.array((0, 255, 255))
        generator = utils_video.generators.add_stimulus_dot(generator, dot_binary, radius=10, center=(20, 20), color=color)
        fly_number = utils.date_to_fly_number(date)
        generator = utils_video.generators.add_text_PIL(generator, f"Fly {fly_number}", (10, stack.shape[1] - 60), size=30)
        generators.append(generator)

    if len(generators) == 0:
        continue
    tile_shape, generators[0] = utils_video.utils.get_generator_shape(generators[0])
    tile_shape = (tile_shape[0], tile_shape[1], 3)
    text_tile = utils_video.generators.video(beh_examples[behaviour]["video_file"], size=(tile_shape[0], -1), start=beh_examples[behaviour]["start"])
    text_tile = utils_video.generators.add_text_PIL(text_tile, utils.rename_behaviour(behaviour), (5, 5), size=40)
    time_stamps = [f"{t:.1f}s" for t in np.arange(-1, 4, 1/100)]
    text_tile = utils_video.generators.add_text_PIL(text_tile, time_stamps, (5, 48), size=40)
    text_tile = utils_video.generators.pad(text_tile, 0, 0, 64, 214)

    generators = [text_tile,] + generators

    grid_generator = utils_video.generators.grid(generators, padding=(15, None), ratio=16/9, allow_different_length=True)
    n_frames = min(500, np.max(n_valid_frames))
    utils_video.make_video(out_path, grid_generator, 100, n_frames=n_frames)
    epoch_info.to_csv(f"epoch_info_{behaviour}.csv")
