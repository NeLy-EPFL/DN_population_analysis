import os
import csv
import glob

from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib
import tempfile
import utils2p
import h5py
import numpy as np

import utils2p


skip_existing = True

directories = glob.glob("/scratch/izar/aymanns/ABO_data_h5/*/*")
for fly_dir in directories:
    model_dir = os.path.join(fly_dir, "models_green_denoising_hdf5")
    models = glob.glob(os.path.join(model_dir, "*/*.h5"))
    if len(models) == 0:
        print(f"No models found for in {fly_dir}")
        continue
    models = sorted(list(models))
    model_path = models[-1]
    print(f"Using model saved in {model_path}.")

    for trial_tif in glob.glob(os.path.join(fly_dir, "warped_green_*_coronal.tif")):
    #for trial_tif in glob.glob(os.path.join(fly_dir, "warped_green_*_coronal*_ref.tif")):
        denoised_tif = os.path.splitext(trial_tif)[0] + "_denoised.tif"
        if skip_existing and os.path.isfile(denoised_tif):
            continue
        
        print(trial_tif + "\n" + "#"*len(trial_tif))
        
        generator_param = {}
        inferrence_param = {}
        
        # We are reusing the data generator for training here. Some parameters like steps_per_epoch are irrelevant but currently needs to be provided
        generator_param["type"] = "generator"
        generator_param["name"] = "SingleTifGenerator"
        generator_param["pre_post_frame"] = 30
        generator_param["pre_post_omission"] = 0
        generator_param["steps_per_epoch"] = 5
        
        generator_param["train_path"] = trial_tif
        
        generator_param["batch_size"] = 5
        generator_param["start_frame"] = 0
        generator_param["end_frame"] = -1  # -1 to go until the end.
        generator_param[
            "randomize"
        ] = 0  # This is important to keep the order and avoid the randomization used during training
        
        
        inferrence_param["type"] = "inferrence"
        inferrence_param["name"] = "core_inferrence"
        
        # Replace this path to where you stored your model
        inferrence_param[
            "model_path"
        ] = model_path
        
        # Replace this path to where you want to store your output file
        inferrence_param[
            "output_file"
        ] = os.path.join(fly_dir, "output.h5")
        
        jobdir = fly_dir
        
        #try:
        #    os.mkdir(jobdir)
        #except:
        #    print("folder already exists")
        
        path_generator = os.path.join(jobdir, "generator.json")
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)
        
        path_infer = os.path.join(jobdir, "inferrence.json")
        json_obj = JsonSaver(inferrence_param)
        json_obj.save_json(path_infer)
        
        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)
        
        inferrence_obj = ClassLoader(path_infer)
        inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)
        
        # Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
        inferrence_class.run()
        
        with h5py.File(inferrence_param["output_file"], "r") as f:
            data = f["data"][()]
            utils2p.save_img(denoised_tif, np.squeeze(data))
