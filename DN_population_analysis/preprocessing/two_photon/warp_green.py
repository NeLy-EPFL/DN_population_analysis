import os.path

import numpy as np
import tqdm

import utils2p
import ofco.warping

import DN_population_analysis.utils as utils

skip_existing = True

f = "../../../recordings.txt"

directories = utils.load_exp_dirs(f)

for directory in directories:
    print(directory + "\n" + "#"*len(directory))
    directory = os.path.join(directory, "2p")
    w_file = os.path.join(directory, "w.npy")
    green_stack_file = os.path.join(directory, "green.tif")
    output_file = os.path.join(directory, "warped_green.tif")
    if skip_existing and os.path.isfile(output_file):
        print(f"Skipping because {output_file} exists.")
        continue
    if not os.path.isfile(w_file):
        print("Skipping because w.npy is missing.")
        continue
    w = np.load(w_file, mmap_mode="r")
    stack = utils2p.load_img(green_stack_file)
    for t in tqdm.tqdm(range(stack.shape[0])):
        stack[t] = ofco.warping.bilinear_interpolate(stack[t], w[t, :, :, 0], w[t, :, :, 1])
    utils2p.save_img(output_file, stack)
