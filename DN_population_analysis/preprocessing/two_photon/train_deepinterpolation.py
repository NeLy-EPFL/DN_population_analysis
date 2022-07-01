import deepinterpolation as de
import sys
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict
import pathlib
import glob
import math
import tifffile
import DN_population_analysis.utils as utils


exp_dirs = utils.load_exp_dirs("../../../recordings.txt")

for fly_dir, trial_dirs in utils.group_by_fly(exp_dirs).items():

    models_dir = os.path.join(fly_dir, "models_green_denoising_hdf5")
    os.makedirs(models_dir, exist_ok=True)
    
    
    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")
    
    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}
    
    # An epoch is defined as the number of batches pulled from the dataset. Because our datasets are VERY large. Often, we cannot
    # go through the entirity of the data so we define an epoch slightly differently than is usual.
    #steps_per_epoch = 10
    pre_post_frame = 30
    batch_size = 20
    start_frame = 0
    end_frame = 500#100
    
    #data_paths = sorted(list(glob.glob(os.path.join(tmpdirname, "*.tif"))))
    data_paths = []
    for trial_dir in trial_dirs:
        data_paths.append(os.path.join(trial_dir, "2p/warped_green.hdf"))
    data_paths =  sorted(data_paths)
    #data_paths = data_paths[5:]
    test_paths = [data_paths[0], data_paths[-1]]
    train_paths = data_paths[1:-1:2]
    steps_per_epoch = math.floor((end_frame - start_frame - 2 * pre_post_frame) * len(train_paths) / batch_size)
    
    generator_param = {}
    generator_param["type"] = "generator"
    generator_param["name"] = "HDF5Generator"
    generator_param["pre_post_frame"] = pre_post_frame
    generator_param["batch_size"] = batch_size
    generator_param["start_frame"] = start_frame
    #generator_param["end_frame"] = end_frame
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["pre_post_omission"] = 0
    
    train_generator_param_list = []
    for indiv_path in train_paths:
        generator_param["train_path"] = indiv_path
        generator_param["end_frame"] = end_frame
        train_generator_param_list.append(generator_param.copy())
    
    test_generator_param_list = []
    for indiv_path in test_paths:
        generator_param["train_path"] = indiv_path
        generator_param["end_frame"] = 100
        test_generator_param_list.append(generator_param.copy())
    
    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param[
        "name"
    ] = "unet_single_1024"  # Name of network topology in the collection
    
    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = run_uid
    training_param["batch_size"] = batch_size
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 25  # network model is potentially saved during training between a regular nb epochs
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 15#9  # if you want to cycle through the entire data. Two many iterations will cause noise overfitting
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = pre_post_frame
    training_param["loss"] = "mean_absolute_error"
    training_param[
        "nb_workers"
    ] = 40  # this is to enable multiple threads for data generator loading. Useful when this is slower than training
    
    training_param["model_string"] = (
        network_param["name"]
        + "_"
        + training_param["loss"]
        + "_"
        + training_param["run_uid"]
    )
    
    # Where do you store ongoing training progress
    jobdir = os.path.join(
        #"/Users/jeromel/test", training_param["model_string"] + "_" + run_uid,
        models_dir, training_param["model_string"] + "_" + run_uid,
    )
    training_param["output_dir"] = jobdir
    
    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")
    
    # Here we create all json files that are fed to the training. This is used for recording purposes as well as input to the
    # training process
    path_training = os.path.join(jobdir, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)
    
    list_train_generator = []
    for local_index, indiv_generator in enumerate(train_generator_param_list):
        path_generator = os.path.join(
                jobdir, "generator" + str(local_index) + ".json")
        json_obj = JsonSaver(indiv_generator)
        json_obj.save_json(path_generator)
        generator_obj = ClassLoader(path_generator)
        train_generator = generator_obj.find_and_build()(path_generator)
        list_train_generator.append(train_generator)
    
    list_test_generator = []
    for local_index, indiv_generator in enumerate(test_generator_param_list):
        path_generator = os.path.join(
                jobdir, "generator" + str(local_index) + ".json")
        json_obj = JsonSaver(indiv_generator)
        json_obj.save_json(path_generator)
        generator_obj = ClassLoader(path_generator)
        test_generator = generator_obj.find_and_build()(path_generator)
        list_test_generator.append(test_generator)
    
    path_network = os.path.join(jobdir, "network.json")
    json_obj = JsonSaver(network_param)
    json_obj.save_json(path_network)
    
    # We find the network obj in the collection using the json file
    network_obj = ClassLoader(path_network)
    
    # We find the training obj in the collection using the json file
    trainer_obj = ClassLoader(path_training)
    
    # We build the generators object. This will, among other things, calculate normalizing parameters.
    global_train_generator = de.generator_collection.CollectorGenerator(list_train_generator)
    global_test_generator = de.generator_collection.CollectorGenerator(list_test_generator)
    print("length of global train generator", len(global_train_generator))
    
    # We build the network object. This will, among other things, calculate normalizing parameters.
    network_callback = network_obj.find_and_build()(path_network)
    
    # We build the training object.
    training_class = trainer_obj.find_and_build()(
        global_train_generator, global_test_generator, network_callback, path_training
    )
    
    # Start training. This can take very long time.
    training_class.run()
    
    # Finalize and save output of the training.
    training_class.finalize()
