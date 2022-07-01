import os.path
import pickle

import numpy as np
import pandas as pd
import scipy.signal

import utils2p
import utils2p.synchronization
import utils_ballrot
import utils_video
import utils_video.generators

import DN_population_analysis.utils as utils


skip_existing = False

trial_dirs = [
    "/mnt/data2/FA/210830_Ci1xG23/Fly1/003_coronal",
    "/mnt/data2/FA/210910_Ci1xG23/Fly2/003_coronal",
    "/mnt/data2/FA/211026_Ci1xG23/Fly3/003_coronal",
    "/mnt/data2/FA/211027_Ci1xG23/Fly2/003_coronal",
    "/mnt/data2/FA/211029_Ci1xG23/Fly1/003_coronal",
]

def make_video_based_on_binary_seq(binary_seq, trial_dir, beh, descriptor):
    if "ref" in trial_dir:
        video_file = os.path.join(trial_dir[:-9], "behData/images/camera_1.mp4")
    else:
        video_file = os.path.join(trial_dir, "behData/images/camera_1.mp4")
    event_based_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(binary_seq) 
    generators = []
    max_length = 0
    for event_number in np.unique(event_numbers):
        if event_number == -1:
            continue
        event_mask = (event_numbers == event_number)
        event_length = np.max(event_based_indices[event_mask]) + 1 
        if event_length > max_length:
            max_length = event_length
        start = np.where(np.logical_and(event_mask, event_based_indices == 0))[0][0] - 200
        generator = utils_video.generators.video(video_file, start=start)
        dot_mask = event_mask[start:]
        dot_mask[:200] = 0
        generator = utils_video.generators.add_stimulus_dot(generator, dot_mask)
        generators.append(generator)

    print("Number of generators:", len(generators))
    print("Max event length:", max_length)
    if len(generators) > 0:
        output_file = f"/mnt/internal_hdd/aymanns/daart_heuristic_videos/{beh}_{descriptor}.mp4"
        grid_generator = utils_video.generators.grid(generators, allow_different_length=True)
        utils_video.make_video(output_file, grid_generator, 100, n_frames=max_length + 200)


filt = scipy.signal.butter(10, 4, btype="lowpass", fs=100, output="sos")

# the state ordering should be the same between the hand and heuristic labels
state_mapping = {
    0: 'background',
    1: 'resting',
    2: 'walking',
    3: 'head_grooming',
    4: 'foreleg_grooming',
    5: 'hind_grooming',
    }

for trial_dir in trial_dirs:
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")
    output_file = f"labels-heuristic/{descriptor}_labels.pkl"
    
    print(output_file)

    if skip_existing and os.path.isfile(output_file):
        continue

    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    df = pd.read_pickle(input_file).filter(like="Pose")

    finite_difference_coefficients = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])
    # step size (spacing between elements)
    h = 0
    kernel = np.zeros(len(finite_difference_coefficients) * (1 + h) - h)
    kernel[::(h+1)] = finite_difference_coefficients

    motion_energy_df = pd.DataFrame()
    direction_df = pd.DataFrame()
    front_height_df = pd.DataFrame()
    for pair in ("F", "M", "H"):
        for side in ("R", "L"):
            for joint in ("Femur", "Tibia", "Tarsus", "Claw"):
                derivatives = []
                for axis in ("x", "y", "z"):
                    col_name = f"Pose__{side}{pair}_leg_{joint}_{axis}"
                    deriv = np.convolve(df[col_name], kernel, mode="same")
                    derivatives.append(deriv) 

                    if pair == "F" and axis == "z" and joint == "Tarsus":
                        front_height_df[f"{side}"] = df[col_name]

                derivatives = np.array(derivatives)
                motion_energy_df[f"{side}{pair}_{joint}"] = scipy.signal.sosfiltfilt(filt, np.sum(np.abs(derivatives), axis=0))
                direction_vectors = np.abs(derivatives) / np.linalg.norm(derivatives, axis=0)
                alpha = np.arctan(direction_vectors[0, :] / direction_vectors[2, :])
                beta = np.arctan(direction_vectors[0, :] / direction_vectors[1, :])
                direction_df[f"{side}{pair}_{joint}_alpha"] = scipy.signal.sosfiltfilt(filt, alpha)
                direction_df[f"{side}{pair}_{joint}_beta"] = scipy.signal.sosfiltfilt(filt, beta)
                direction_df[f"{side}{pair}_{joint}_x"] = derivatives[0, :]
                direction_df[f"{side}{pair}_{joint}_y"] = derivatives[1, :]
                direction_df[f"{side}{pair}_{joint}_z"] = derivatives[2, :]

    total_motion_energy = np.sum(motion_energy_df.values, axis=1)
    front_motion_energy = np.sum(motion_energy_df.filter(like="F_"), axis=1)
    hind_motion_energy = np.sum(motion_energy_df.filter(like="H_"), axis=1)

    front_height_df["height"] = scipy.signal.sosfiltfilt(filt, (front_height_df["R"] + front_height_df["L"]) / 2)

    fictrac_file = utils2p.find_fictrac_file(trial_dir, most_recent=True)
    fictrac_data = utils_ballrot.load_fictrac(fictrac_file)

    vel = fictrac_data["delta_rot_forward"]
    filtered_vel = scipy.signal.sosfiltfilt(filt, vel)
    
    binary_seq_walking = (filtered_vel > 0.5)
    binary_seq_walking = utils.utils.hysteresis_filter(binary_seq_walking, n=100, n_false=50)
    
    binary_seq_remaining = ~binary_seq_walking
    binary_seq_resting = np.logical_and(binary_seq_remaining, total_motion_energy < 0.3)
    binary_seq_resting = utils.hysteresis_filter(binary_seq_resting, n=50)
    
    binary_seq_remaining = np.logical_and(binary_seq_remaining, ~binary_seq_resting)
    binary_seq_grooming = np.logical_and(front_motion_energy > 0.2, hind_motion_energy < 0.2)
    binary_seq_grooming = np.logical_and(binary_seq_remaining, binary_seq_grooming)
    binary_seq_grooming = utils.hysteresis_filter(binary_seq_grooming, n=50)

    binary_seq_eye_grooming = np.logical_and(binary_seq_grooming, front_height_df["height"].values > 0.05)
    binary_seq_eye_grooming = utils.hysteresis_filter(binary_seq_eye_grooming, n=50)
    
    binary_seq_foreleg_grooming = np.logical_and(binary_seq_grooming, front_height_df["height"].values < 0.05)
    binary_seq_foreleg_grooming = utils.hysteresis_filter(binary_seq_foreleg_grooming, n=50)

    binary_seq_remaining = np.logical_and(binary_seq_remaining, ~binary_seq_grooming)
    binary_seq_hindgrooming = np.logical_and(front_motion_energy < 0.2, hind_motion_energy > 0.2)
    binary_seq_hindgrooming = np.logical_and(binary_seq_remaining, binary_seq_hindgrooming)
    binary_seq_hindgrooming = utils.hysteresis_filter(binary_seq_hindgrooming, n=50)
    
    states = np.zeros(df.shape[0], dtype=int)

    states[binary_seq_resting] = 1
    states[binary_seq_walking] = 2
    states[binary_seq_eye_grooming] = 3
    states[binary_seq_foreleg_grooming] = 4
    states[binary_seq_hindgrooming] = 5

    data = {'states': states, 'state_labels': state_mapping}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
