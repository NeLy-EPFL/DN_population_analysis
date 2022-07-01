import os
import glob
import pickle

import numpy as np
import pandas as pd
import behavelet

import df3dPostProcessing
import df3dPostProcessing.utils.utils_plots
import utils_video
import utils_video.generators

import DN_population_analysis.utils as utils


update = False
make_videos = False
output_file = "beh_data.pkl"
folders_without_df3d = ""
calc_wavelets = False
skip_existing = False

if update and os.path.isfile(output_file):
    data = pd.read_pickle(output_file)
else:
    multi_index = pd.MultiIndex(
        levels=[[]] * 5,
        codes=[[]] * 5,
        names=["Date", "Genotype", "Fly", "Trial", "Frame"],
    )
    data = pd.DataFrame(index=multi_index)

f = "../../../recordings.txt"
directories = utils.load_exp_dirs(f)

for i, directory in enumerate(directories):
    print(directory + "\n" + "#" * len(directory))
    date = utils.get_date(directory)
    genotype = utils.get_genotype(directory)
    fly = utils.get_fly_number(directory)
    trial = utils.get_trial_number(directory)

    if len(data.index) > 0 and data.index.isin([(date, genotype, fly, trial, 0)]).any():
        print("Skipped because it already exists in df.")
        continue

    if os.path.isdir(os.path.join(directory, "behData/images/df3d_new")):
        output_subfolder = "df3d_new"
    elif os.path.isdir(os.path.join(directory, "behData/images/df3d_mm")):
        output_subfolder = "df3d_mm"
    elif os.path.isdir(os.path.join(directory, "behData/images/df3d")):
        output_subfolder = "df3d"
    else:
        folders_without_df3d += directory + "\n"
        print("skipping because there is no df3d output folder")
        continue
    if output_subfolder == "df3d_new":
        possible_pose_results = glob.glob(
            os.path.join(directory, f"behData/images/{output_subfolder}/df3d_result*.pkl")
        )
    else:
        possible_pose_results = glob.glob(
            os.path.join(directory, f"behData/images/{output_subfolder}/pose_result*.pkl")
        )

    if len(possible_pose_results) == 0:
        folders_without_df3d += directory + "\n"
        print("skipping because there is no pose result file")
        continue

    trial_data_file = os.path.join(
        directory, f"behData/images/{output_subfolder}/post_processed.pkl"
    )
    if not skip_existing or not os.path.isfile(trial_data_file):

        change_times = [os.stat(path).st_mtime for path in possible_pose_results]
        most_recent_pose_result = possible_pose_results[np.argmax(change_times)]

        df3dPost = df3dPostProcessing.df3dPostProcess(
            most_recent_pose_result, calculate_3d=True, outlier_correction=True
        )
        #df3dPost = df3dPostProcessing.df3dPostProcess(most_recent_pose_result)

        aligned = df3dPost.align_to_template(all_body=True)
        if make_videos:
            generator = utils_video.generators.df3d_line_plots_aligned(aligned)
            pose_video = os.path.join(directory, f"behData/processed_pose.mp4")
            utils_video.make_video(pose_video, generator, 100)
        angles = df3dPost.calculate_leg_angles()

        n_frames = len(aligned["LF_leg"]["Femur"]["raw_pos_aligned"])
        frames = np.arange(n_frames)

        trial_indices = pd.MultiIndex.from_arrays(
            [
                np.array([date,] * n_frames),
                np.array([genotype,] * n_frames),
                np.array([fly,] * n_frames),
                np.array([trial,] * n_frames),
                frames,
            ],
            names=("Date", "Genotype", "Fly", "Trial", "Frame")
        )
        trial_data = pd.DataFrame(index=trial_indices)

        for leg_name, leg_data in aligned.items():
            for joint_name, joint_data in leg_data.items():
                if joint_name == "Coxa":
                    key = "fixed_pos_aligned"
                    joint_data[key] = np.tile(joint_data[key], (n_frames, 1))
                else:
                    key = "raw_pos_aligned"
                for i, axes in enumerate(["x", "y", "z"]):
                    col_name = "_".join(["Pose_", leg_name, joint_name, axes])
                    trial_data[col_name] = joint_data[key][:, i]

        for leg_name, leg_data in angles.items():
            for angle_name, angle_data in leg_data.items():
                col_name = "_".join(["Angle_", leg_name, angle_name])
                trial_data[col_name] = angle_data

        print(trial_data.shape)
        trial_data.to_pickle(trial_data_file, protocol=4)
    else:
        print("loading trial data from file because it already exists")

    angle_wavelet_file = os.path.join(
        directory, f"behData/images/{output_subfolder}/angle_wavelets.pkl"
    )
    if calc_wavelets and (not os.path.isfile(angle_wavelet_file) or not skip_existing):
        trial_data = pd.read_pickle(trial_data_file)
        angle_data = trial_data.filter(regex="Angle")
        columns = angle_data.columns
        X = angle_data.values
        freqs, power, X_coeff = behavelet.wavelet_transform(
            X, n_freqs=25, fsample=100.0, fmin=1.0, fmax=50.0, gpu=True
        )
        coeff_columns = [f"Coeff {col} {freq}Hz" for col in columns for freq in freqs]
        n_frames = len(X_coeff)
        frames = np.arange(n_frames)
        indices = trial_data.index
        trial_wavelets = pd.DataFrame(index=indices)
        trial_wavelets[coeff_columns] = X_coeff
        trial_wavelets.to_pickle(angle_wavelet_file)

    data = data.append(trial_data)

    data.to_pickle(output_file)

with open("folders_without_df3d.txt", "w") as f:
    f.write(folders_without_df3d)
