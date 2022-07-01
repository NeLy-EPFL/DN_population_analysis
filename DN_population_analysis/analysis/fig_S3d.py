###
Note that the data necess....
###

import glob
import copy
import os.path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm
import mpl_toolkits.axes_grid1
import utils2p

import utils


def remove_axis(ax):
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

h_space = 10
##cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
#cmap_array = matplotlib.cm.jet(np.arange(256))
##cmap = matplotlib.colors.ListedColormap(cmap_array[:900], N=900)
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", cmap_array[:230], N=256)
#cmap.set_bad("white", 1.)

ref_trials = ["/mnt/data2/FA/210830_Ci1xG23/Fly1/001_coronal_001_ref/",
              "/mnt/data2/FA/210910_Ci1xG23/Fly2/001_coronal_001_ref/", 
              "/mnt/data2/FA/211026_Ci1xG23/Fly3/001_coronal_001_ref/", 
              "/mnt/data2/FA/211027_Ci1xG23/Fly2/001_coronal_001_ref/", 
              "/mnt/data2/FA/211029_Ci1xG23/Fly1/001_coronal_001_ref/",
             ]

frame_indices = {"all_resting": [72, 44, 70, 42, 79],
                 "walking": [69, 79, 79, 79, 79],
                 "head_grooming": [61, 34, 39, 30, 65],
                 "foreleg_grooming": [24, 29, 29, 24, 31],
                 "hind_grooming": [42, 24, 25, 32, 34],
                }

if __name__ == "__main__":
    fig, axes = plt.subplots(6, 6, figsize=(4, 2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.1]})
    for ax, beh in zip(axes, ["ref", "all_resting", "walking", "head_grooming", "foreleg_grooming", "hind_grooming"]):
        if beh == "ref":
            for fly_idx, trial_dir in enumerate(ref_trials):
                stack_file = os.path.join(trial_dir, "2p/warped_green.tif")
                stack = utils2p.load_img(stack_file, memmap=True)
                off_set = fly_idx * (640 + h_space)
                mean_img = np.mean(stack, axis=0)
                mean_img, size_corrections = utils.fix_image_size(mean_img, already_cropped=False, trial_dir=trial_dir, return_corrections=True)
                vmin = np.nanpercentile(mean_img, 1)
                vmax = np.nanpercentile(mean_img, 99)
                ax[fly_idx].imshow(mean_img, cmap="Greys", interpolation=None, vmin=vmin, vmax=vmax)
                remove_axis(ax[fly_idx])
                utils.draw_rois(trial_dir, ax[fly_idx],
                                size_corrections[1],
                                size_corrections[0])
            ax[-1].axis("off")
        else:
            frames = []
            for fly_idx, stack_file in enumerate(sorted(glob.glob(f"../videos/*_{beh}.tif"))):
                print(stack_file)
                stack = utils2p.load_img(stack_file, memmap=True)
                off_set = fly_idx * (640 + h_space)
                #frame_idx = 48
                #frame = stack[frame_idx]
                #while np.sum(np.isnan(frame)) / frame.shape[0] / frame.shape[1] > 0.5:
                #    frame_idx = frame_idx - 1
                #    frame = stack[frame_idx]
                frame = stack[frame_indices[beh][fly_idx]]
                frames.append(np.copy(frame))
            vmax = np.nanpercentile(np.array(frames), 99)
            for fly_idx, frame in enumerate(frames):
                im = ax[fly_idx].imshow(frame, cmap=utils.cmap, interpolation=None, vmin=0, vmax=vmax)
                remove_axis(ax[fly_idx])
                if fly_idx == 0:
                    beh = utils.rename_behaviour(beh)
                    ax[fly_idx].set_ylabel(beh, color=utils.behaviour_colours[beh], rotation=0, va="center", ha="right")
            cbar = fig.colorbar(im, cax=ax[-1])
            cbar.ax.set_ylabel(r'%$\frac{\Delta F}{F}$', rotation=0, va="center", ha="left")
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig("event_triggered_averaging.pdf", transparent=True, dpi=1200)
