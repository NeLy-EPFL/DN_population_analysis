import sys
import os
import glob

from skimage import io
import numpy as np

import utils2p

from ofco import motion_compensate
from ofco.utils import default_parameters
import DN_population_analysis.utils as utils


def reassamble_warped_images(folder):
    n_files = len(glob.glob(os.path.join(folder, f"warped_red_*{tag}.tif")))
    stacks = []
    for i in range(n_files):
        substack = utils2p.load_img(os.path.join(folder, f"warped_red_{i}{tag}.tif"))
        stacks.append(substack)
    stack = np.concatenate(stacks, axis=0)
    utils2p.save_img(os.path.join(folder, f"warped_red{tag}.tif"), stack)
    for i in range(n_files):
        path = os.path.join(folder, f"warped_red_{i}{tag}.tif")
        os.remove(path)

def reassamble_vector_fields(folder):
    n_files = len(glob.glob(os.path.join(folder, f"w_*{tag}.npy")))
    vector_fields = []
    for i in range(n_files):
        path = os.path.join(folder, f"w_{i}{tag}.npy")
        sub_fields = np.load(path)
        print(path, sub_fields.shape)
        vector_fields.append(sub_fields)
    vector_field = np.concatenate(vector_fields, axis=0)
    np.save(os.path.join(folder, f"w{tag}.npy"), vector_field)
    for i in range(n_files):
        path = os.path.join(folder, f"w_{i}{tag}.npy")
        os.remove(path)

if __name__ == "__main__":

    for folder in utils.load_exp_dirs("../../../recordings.txt"):
        ref_frame = os.path.join(folder, "2p/ref_frame.tif")
        print("Folder:", folder)
        print("Reference frame:", ref_frame)
        ref_frame = io.imread(ref_frame)
        
        param = default_parameters()
        tag = ""
        for i, substack in enumerate(utils2p.load_stack_batches(os.path.join(folder, "2p/denoised_red.tif"), 28)):
            print(i)
            frames = range(len(substack))
            warped_output = os.path.join(folder, f"2p/warped_red_{i}{tag}.tif")
            w_output = os.path.join(folder, f"2p/w_{i}{tag}.npy")
            if os.path.isfile(warped_output) and os.path.isfile(w_output):
                print("skipped because it exists")
                continue
            stack1_warped, stack2_warped = motion_compensate(
                substack, None, frames, param, parallel=True, verbose=True, w_output=w_output, ref_frame=ref_frame
            )
        
            io.imsave(warped_output, stack1_warped)
        
        reassamble_warped_images(os.path.join(folder, "2p"))
        reassamble_vector_fields(os.path.join(folder, "2p"))
