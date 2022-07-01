import os.path
import csv

import pandas as pd
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt

import utils2p
import utils2p.synchronization

import utils


patch_height = 3
patch_width = 2
#patch_height = 30
#patch_width = 20
#patch_height = 7
#patch_width = 5

skip_existing = False

multi_index = pd.MultiIndex(levels=[[]] * 5, codes=[[]]  * 5, names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
df = pd.DataFrame(index=multi_index)

f = "trials_for_paper_trial_ref.txt"
directories = utils.load_exp_dirs(f)

for trial_dir in directories:
    print(trial_dir + "\n" + "#"*len(trial_dir))
    output_file = os.path.join(trial_dir, "2p/roi_dFF.pkl")
    if skip_existing and os.path.isfile(output_file):
        print("skipping because file exists")
        continue
    date = utils.get_date(trial_dir)
    genotype = utils.get_genotype(trial_dir)
    fly = utils.get_fly_number(trial_dir)
    trial = utils.get_trial_number(trial_dir)
    fly_dir = utils.get_fly_dir(trial_dir)


    #centers_file = f"/mnt/internal_hdd/aymanns/ABO_maxima_density/{date}_{trial:03}_MAX_denoised_green.txt"
    #centers_file = os.path.join(fly_dir, "roi_centers.txt")
    centers_file = os.path.join(trial_dir, "2p/roi_centers.txt")

    if not os.path.exists(centers_file):
        print(f"skipping because {centers_file} does not exist")
        continue
    centers = np.loadtxt(centers_file)

    df = pd.DataFrame(index=multi_index)
    for name, source in [("warped", "warped_green.tif"), ("denoised", "denoised_green.tif"),]:# ("dFF", "dFF.tif")]:
        print(name)
        if "ref" in trial_dir:
            sync_file = utils2p.find_sync_file(trial_dir[:-9])
            capture_json = utils2p.find_seven_camera_metadata_file(trial_dir[:-9])
            metadata_file = utils2p.find_metadata_file(trial_dir[:-9])
            sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir[:-9])
        else:
            sync_file = utils2p.find_sync_file(trial_dir)
            capture_json = utils2p.find_seven_camera_metadata_file(trial_dir)
            metadata_file = utils2p.find_metadata_file(trial_dir)
            sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)

        cam_line, frame_counter = utils2p.synchronization.get_lines_from_h5_file(sync_file, ("Cameras", "Frame Counter"))
        metadata = utils2p.Metadata(metadata_file)
        frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata)
        cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
        
        if name in ("denoised", "dFF"):
            denoising_kernel_size = 30
        else:
            denoising_kernel_size = 0
        mask = np.logical_and(frame_counter >= denoising_kernel_size, cam_line >= 0)
        mask = np.logical_and(mask, frame_counter <= np.max(frame_counter) - denoising_kernel_size)
        cam_line, frame_counter = utils2p.synchronization.crop_lines(mask, (cam_line, frame_counter))
        frame_counter = frame_counter - denoising_kernel_size

        sync_metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
        freq = sync_metadata.get_freq()
        times = utils2p.synchronization.get_times(len(cam_line), freq)
        cam_frame_indices = utils2p.synchronization.get_start_times(cam_line, cam_line, zero_based_counter=True)
        frame_times = utils2p.synchronization.get_start_times(cam_line, times, zero_based_counter=True)
        frame_times_2p = utils2p.synchronization.get_start_times(frame_counter, times, zero_based_counter=True)

        stack_file = os.path.join(trial_dir, f"2p/{source}")
        stack = utils2p.load_img(stack_file)

        overall_value = np.sum(stack, axis=(1, 2))
        for zero_frame in np.where(overall_value < 1e-5)[0]:
            stack[zero_frame] = (stack[zero_frame - 1] + stack[zero_frame + 1]) / 2

        if name == "warped":
            #with open(os.path.join(fly_dir, "crop_parameters.csv")) as csv_file:
            with open(os.path.join(trial_dir, "2p/crop_parameters.csv")) as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
                _, _, _, _ = next(reader)
                height_offset, height, width_offset, width = tuple(map(int, next(reader)))
            stack = stack[:, height_offset : height_offset + height, width_offset : width_offset + width]

        for roi, center in enumerate(centers):
            print(f"\t{roi}")
            start_height = int(max(center[1] - patch_height, 0))
            start_width = int(max(center[0] - patch_width, 0))
            stop_height = int(min(center[1] + patch_height, stack.shape[1]))
            stop_width = int(min(center[0] + patch_width, stack.shape[0]))

            patch = stack[:, start_height : stop_height, start_width : stop_width]

            if name == "dFF":
                dFF = np.mean(patch, axis=(1, 2))
            else:
                F = np.mean(patch, axis=(1, 2))
                kernel_length = 15
                kernel = np.ones(kernel_length) / kernel_length
                F_0 = np.min(np.convolve(F, kernel, mode="valid"))
                dFF = (F - F_0) / F_0 * 100

            # Interpolation of NaNs
            nans = np.logical_or(np.isnan(dFF), np.isinf(dFF))
            if sum(nans) == len(nans):
                continue
            if sum(nans) > 0:
                x = np.arange(len(dFF))
                interp_locations = x[nans]
                value_locations = x[~nans]
                values = dFF[~nans]
                dFF[nans] = np.interp(interp_locations, value_locations, values)
            # Interpolation to match behaviour frames
            dFF = dFF[np.min(frame_counter) : np.max(frame_counter) + 1]
            interpolator = scipy.interpolate.interp1d(frame_times_2p, dFF, kind="linear", bounds_error=False, fill_value="extrapolate")
            dFF = interpolator(frame_times)

            # Plot trace
            os.makedirs(os.path.join(trial_dir, "traces"), exist_ok=True)
            plt.figure(figsize=(14, 4))
            plt.plot(frame_times, dFF)
            plt.xlabel("Time [s]")
            plt.ylabel("%dF/F")
            plt.savefig(os.path.join(trial_dir, f"traces/{name}_roi_{roi}.pdf"))
            plt.close()

            frames = cam_frame_indices 
            n_frames = len(frames)

            indices = pd.MultiIndex.from_arrays(([date, ] * n_frames,
                                                 [genotype, ] * n_frames,
                                                 [fly, ] * n_frames,
                                                 [trial, ] * n_frames,
                                                 frames,
                                                ),
                                                names=[u'Date', u'Genotype', u'Fly', u'Trial', u'Frame'])
            roi_df = pd.DataFrame(index=indices)
            roi_df["dFF"] = dFF
            roi_df["ROI"] = roi
            roi_df["Source file"] = name
            roi_df["Time"] = frame_times

            df = df.append(roi_df)
    df.to_pickle(output_file)

#beh_df = pd.read_pickle("/mnt/internal_hdd/aymanns/ABO_data_processing/behaviour_downsampled_to_2p.pkl")
#pose_columns = [col for col in beh_df.columns if "Pose" in col]
#angle_columns = [col for col in beh_df.columns if "Angle" in col]
#beh_df = beh_df.drop(columns=pose_columns)
#beh_df = beh_df.drop(columns=angle_columns)
#
#merged = pd.merge(df, beh_df,
#                  how="left",
#                  left_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
#                  right_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
#                  left_index=True,
#                  right_index=True)
#
## The behaviour capture stops earlier than the 2p capture.
## This line deletes the superfluous 2p frames.
#merged = merged.dropna()
#
#merged.to_csv(f"/mnt/internal_hdd/aymanns/ABO_data_processing/beh_centers_{patch_height}x{patch_width}_dFF.csv")
#merged.to_pickle(f"/mnt/internal_hdd/aymanns/ABO_data_processing/beh_centers_{patch_height}x{patch_width}_dFF.pkl")
