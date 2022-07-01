import os.path

import matplotlib.pyplot as plt
import numpy as np
import utils2p

import DN_population_analysis.utils as utils


trial_dirs = sorted(utils.load_exp_dirs("../../recordings.txt"))
trial_dirs = [d for d in trial_dirs if utils.get_trial_number(d) == 1]

fig, ax = plt.subplots(1, 5, figsize=(8, 1))

for i, trial_dir in enumerate(trial_dirs):

    mean_img = utils2p.load_img(os.path.join(trial_dir, "2p/mean_green.tif"))
    mean_img, size_corrections = utils.fix_image_size(mean_img, already_cropped=False, trial_dir=trial_dir, return_corrections=True)
    vmin = np.nanpercentile(mean_img, 1)
    vmax = np.nanpercentile(mean_img, 99)
    ax[i].imshow(mean_img, cmap="Greys", interpolation=None, vmin=vmin, vmax=vmax)
    utils.remove_axis(ax[i])
    utils.draw_rois(trial_dir, ax[i],
                    size_corrections[1],
                    size_corrections[0])

plt.savefig("output/Fig2/panel_d.pdf", dpi=3000)
