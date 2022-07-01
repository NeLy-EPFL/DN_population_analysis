import os.path
import math
import csv

import numpy as np
import scipy.ndimage

import utils2p
from utils import *

def find_pixel_wise_baseline(stack, n=10, occlusions=None):
    """
    This functions finds the indices of n consecutive frames that can serve
    as a fluorescence baseline. It convolves the fluorescence trace of each
    pixel with a rectangular signal of length n and finds the
    minimum of the convolved signal.

    Parameters
    ----------
    stack : np.array 3D
        First dimension should encode time.
        Second and third dimension are for space.
    n : int, default = 10
        Length of baseline.
    occlusions : numpy array of type boolean
        Occlusions are ignored in baseline calculation.
        Default is None.

    Returns
    -------
    baseline_img : np.array 2D
        Baseline image.
    """
    convolved = scipy.ndimage.convolve1d(stack, np.ones(n), axis=0)
    if occlusions is not None:
        occ_convolved = scipy.ndimage.convolve1d(occlusions, np.ones(n), axis=0)
        occluded_pixels = np.where(occ_convolved)
        if np.issubdtype(convolved.dtype, np.integer):
            convolved[occluded_pixels] = np.iinfo(convolved.dtype).max
        elif np.issubdtype(convolved.dtype, np.floating):
            convolved[occluded_pixels] = np.finfo(convolved.dtype).max
        else:
            raise RuntimeError(f"Data type of stack and occluded has to be float or int not {convolved.dtype}")
    length_of_valid_convolution = max(stack.shape[0], n) - min(stack.shape[0], n) + 1
    start_of_valid_convolution = math.floor(n / 2)
    convolved = convolved[
        start_of_valid_convolution : start_of_valid_convolution
        + length_of_valid_convolution
    ]
    indices = np.argmin(convolved, axis=0)
    baseline_img = np.zeros(stack.shape[1:])
    for i in range(stack.shape[1]):
        for j in range(stack.shape[2]):
            baseline_i_j = np.arange(indices[i, j], indices[i, j] + n)
            baseline_img[i, j] = np.sum(stack[baseline_i_j, i, j]) / n
    return baseline_img


def interpolate_zero_locations(stack, path):
    zero_locations = np.where(np.isclose(stack, 0, atol=1e-10))
    effected_frames = np.unique(zero_locations[0])
    if len(effected_frames) == 0:
        return stack
    print("Effected frames:", effected_frames)
    
    for frame in effected_frames:
        mask = zero_locations[0] == frame
        frame_zero_locations = (zero_locations[0][mask], zero_locations[1][mask], zero_locations[2][mask])
    
        neg_increment = 1
        while frame - neg_increment in effected_frames:
            neg_increment += 1
        pos_increment = 1
        while frame + pos_increment in effected_frames:
            pos_increment += 1
    
        previous_frame_locations = (frame_zero_locations[0] - neg_increment, frame_zero_locations[1], frame_zero_locations[2])
        next_frame_locations = (frame_zero_locations[0] + pos_increment, frame_zero_locations[1], frame_zero_locations[2])
    
        if frame - neg_increment < 0 and frame + pos_increment > len(stack) - 1:
            raise ValueError("All frames are effected by zeros.")
        elif frame - neg_increment < 0:
            stack[frame_zero_locations] = stack[next_frame_locations]
        elif frame + pos_increment > len(stack) - 1:
            stack[frame_zero_locations] = stack[previous_frame_locations]
        else:
            stack[frame_zero_locations] = stack[previous_frame_locations] + (stack[next_frame_locations] - stack[previous_frame_locations]) / (neg_increment + pos_increment) * neg_increment
    
    utils2p.save_img(path, stack)
    return stack



skip_existing_baseline = True
skip_existing_dFF = True
f = "../../recordings.txt"

directories = load_exp_dirs(f)

# Denoised
for fly_dir, trial_dirs in group_by_fly(directories).items():
    print(fly_dir + "\n" + "#"*len(fly_dir))
    baseline_imgs = []
    #with open(os.path.join(fly_dir, "crop_parameters.csv")) as csv_file:
    #    reader = csv.reader(csv_file, delimiter=",")
    #    _, _, _, _ = next(reader)
    #    height_offset, height, width_offset, width = tuple(map(int, next(reader)))
    fly_baseline_file = os.path.join(fly_dir, "dFF_baseline_denoised.tif")
    if False:
        pass
    #if skip_existing_baseline and os.path.isfile(fly_baseline_file):
    #    print(f"skipping baseline calculation because {fly_baseline_file} exists")
    #    fly_baseline_img = utils2p.load_img(fly_baseline_file)
    else:
        for trial_dir in trial_dirs:
            print(trial_dir)
            #stack_file = os.path.join(trial_dir, "2p/denoised_green_izar_9_epochs.tif")
            stack_file = os.path.join(trial_dir, "2p/denoised_green.tif")
            if not os.path.isfile(stack_file):
                print(f"Skipping because {stack_file} does not exist.")
                continue
            baseline_file = os.path.join(trial_dir, "2p/trial_dFF_baseline_denoised.tif")
            if skip_existing_baseline and os.path.isfile(baseline_file):
                baseline_img = utils2p.load_img(baseline_file)
            else:
                stack = utils2p.load_img(stack_file)#[:, height_offset : height_offset + height, width_offset : width_offset + width]
                stack = interpolate_zero_locations(stack, stack_file)
                occlusion = np.isclose(stack, 0, atol=1e-10)
                #filtered_stack = scipy.ndimage.filters.median_filter(stack, (3, 3, 3))
                #filtered_stack = scipy.ndimage.filters.gaussian_filter(stack, (0, 3, 3))
                filtered_stack = stack
                baseline_img = find_pixel_wise_baseline(filtered_stack, n=15, occlusions=occlusion.astype(np.int))
                utils2p.save_img(baseline_file, baseline_img)
                del stack, filtered_stack
            baseline_imgs.append(baseline_img)
        fly_baseline_img = np.min(np.array(baseline_imgs), axis=0)
        #utils2p.save_img(fly_baseline_file, fly_baseline_img)
    print("Calculating dFF")
    for trial_dir in trial_dirs:
        print(trial_dir)
        #dff_file = os.path.join(trial_dir, "2p/dFF_denoised.tif")
        dff_file = os.path.join(trial_dir, "2p/dFF.tif")
        if skip_existing_dFF and os.path.isfile(dff_file):
            print(f"skipping because {dff_file} exists")
            continue

        #baseline_img = fly_baseline_img
        baseline_img = utils2p.load_img(os.path.join(trial_dir, "2p/trial_dFF_baseline_denoised.tif"))
        baseline_img = scipy.ndimage.gaussian_filter(baseline_img, 5)

        #stack_file = os.path.join(trial_dir, "2p/denoised_green_izar_9_epochs.tif")
        stack_file = os.path.join(trial_dir, "2p/denoised_green.tif")
        if not os.path.isfile(stack_file):
            print(f"Skipping because {stack_file} does not exist.")
            continue
        stack = utils2p.load_img(stack_file)#[:, height_offset : height_offset + height, width_offset : width_offset + width]
        min_baseline = np.min(baseline_img)
        if min_baseline < 0:
            stack -= min_baseline
            baseline_img -= min_baseline
        dff_stack = (np.divide((stack - baseline_img),
                               baseline_img,
                               out=np.zeros_like(stack, dtype=np.double),
                               where=~np.isclose(baseline_img, 0, atol=1e-10),
                              )
                     * 100
                    )
        utils2p.save_img(dff_file, dff_stack)
        del stack, dff_stack
