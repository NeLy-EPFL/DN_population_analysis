import os
import csv
import shutil

import h5py
import numpy as np

import utils2p
import DN_population_analysis.utils as utils


f = "../../../recordings.txt"
directories = utils.load_exp_dirs(f)
skip_existing = True

for trial_dir in directories:
    print(trial_dir)
        
    with open(os.path.join(trial_dir, "2p/crop_parameters.csv")) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        _, _, _, _ = next(reader)
        height_offset, height, width_offset, width = tuple(map(int, next(reader)))

    stack_file = os.path.join(trial_dir, "2p/warped_green.tif")
    if not os.path.isfile(stack_file):
        print(f"Skipping because {stack_file} does not exist.")
        continue
    stack = utils2p.load_img(stack_file)
    stack = stack[:, height_offset : height_offset + height, width_offset : width_offset + width]

    with h5py.File(os.path.join(trial_dir, f"2p/warped_green.hdf"), "w") as outfile:
        dset = outfile.create_dataset("warped_green", data=stack, chunks=(1, stack.shape[1], stack.shape[2]))
