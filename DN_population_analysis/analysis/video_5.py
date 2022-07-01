import os.path
import glob

import numpy as np
import cv2
import utils2p
import utils2p.synchronization
import utils_video
import utils_video.generators

import DN_population_analysis.utils as utils


def roi_annotation_func(img):
    angle = 0
    color = (255, 255, 255)
    thickness = 2

    dash = 20
    gap = 30
    
    for start_angle in np.arange(0, 360, dash + gap):
        end_angle = start_angle + dash

        center_coordinates = (291, 163)
        axes_length = (13, 11)
        img = cv2.ellipse(img, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness)
        
        center_coordinates = (448, 141)
        axes_length = (11, 16)
        img = cv2.ellipse(img, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness)

    return img

left_touch = {"date": 210910, "genotype": "Ci1xG23", "fly": 2, "trial": 1, "start": 388, "stop": 391}
both_touch = {"date": 210910, "genotype": "Ci1xG23", "fly": 2, "trial": 1, "start": 125, "stop": 128}
right_touch = {"date": 210910, "genotype": "Ci1xG23", "fly": 2, "trial": 3, "start": 448, "stop": 451}

example_generators = []
for example in (left_touch, both_touch, right_touch):

    date = example["date"]
    genotype = example["genotype"]
    fly = example["fly"]
    trial = example["trial"]
    start = example["start"]
    stop = example["stop"]

    trial_dir = utils.get_trial_dir(date, genotype, fly, trial)
    
    beh_video_file = os.path.join(trial_dir, "behData/images/camera_2.mp4")
    nmf_img_dirs = glob.glob(os.path.join(trial_dir, f"behData/images/df3d/nmf/simulation_results_new_sdf/images_{start}00_{stop}00_*"))
    nmf_img_dir = sorted(nmf_img_dirs)[-1]
    nmf_images = os.path.join(nmf_img_dir, "*.png")

    sync_file = utils2p.find_sync_file(trial_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)

    metadata = utils2p.Metadata(metadata_file)

    cam_line, frame_counter = utils2p.synchronization.get_lines_from_sync_file(sync_file, ["Cameras", "Frame Counter"])
    cam_line = utils2p.synchronization.process_cam_line(cam_line, seven_camera_metadata_file)
    frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata=metadata)

    frame_counter = frame_counter - 30

    nmf_generator = utils_video.generators.images(nmf_images)
    beh_generator = utils_video.generators.video(beh_video_file, start=start * 100, stop=stop * 100)
    beh_generator = utils_video.generators.stack([nmf_generator, beh_generator], axis=1, padding=10)

    dff = utils2p.load_img(os.path.join(trial_dir, "2p/dFF.tif"), memmap=True)
    dff_generator = utils_video.generators.dff(dff, vmin=15, annotation_func=roi_annotation_func)
    resample_indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(start * 100, stop * 100), cam_line, frame_counter)
    dff_generator = utils_video.generators.resample(dff_generator, resample_indices)

    generator = utils_video.generators.stack([beh_generator, dff_generator], axis=1, padding=10)
    example_generators.append(generator)

generator = utils_video.generators.stack(example_generators, axis=0, padding=10)
utils_video.make_video("asymmetric_grooming.mp4", generator, fps=25)
