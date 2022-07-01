import os.path

import utils2p
import numpy as np

import DN_population_analysis.utils as utils


for trial_dir in utils.load_exp_dirs("../../../recordings.txt"):
    stack = utils2p.load_img(os.path.join(trial_dir, "2p/warped_green.tif"))
    mean = np.mean(stack, axis=0)
    utils2p.save_img(os.path.join(trial_dir, "2p/mean_green.tif"), mean)
