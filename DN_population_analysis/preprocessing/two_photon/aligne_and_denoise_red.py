import os
import subprocess
import sys
import time
import io

import pandas as pd
import noisy2way

import utils2p
import DN_population_analysis.utils as utils


def aligne_and_denoise(directory, gpu_id):
    red = utils2p.load_img(os.path.join(directory, "red.tif"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    aligned, shift = noisy2way.correct_2way_alignment(
        red,
        basedir=directory,
        model_name="model_red",
        save_denoised_image=os.path.join(directory, "denoised_red.tif"),
    )
    del red
    if shift != 0:
        green = utils2p.load_img(os.path.join(directory, "green.tif"))
        aligned = noisy2way.apply_shift(green, shift)
        utils2p.save_img(os.path.join(directory, "green.tif"), aligned)

if __name__ == "__main__":
    f = "../../../recordings.txt"
    skip_existing = True

    directories = utils.load_exp_dirs(f)

    for directory in directories:
        print("\n" + directory + "\n" + len(directory) * "#")
        if skip_existing and os.path.isfile(
            os.path.join(directory, "2p/denoised_red.tif")
        ):
            print("skipped because denoised_red.tif exists.")
            continue
        aligne_and_denoise(os.path.join(directory, "2p"), 0)
