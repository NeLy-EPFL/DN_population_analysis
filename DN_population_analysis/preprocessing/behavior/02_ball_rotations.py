import os.path
import subprocess

import utils_ballrot

import DN_population_analysis.utils as utils


def fictrac_processing(images_dir):
    fictrac_video_file = os.path.join(images_dir, f"camera_{fictrac_cam}.mp4")

    if os.path.isfile(fictrac_video_file):
        config_file = os.path.join(images_dir, "config.txt")

        if not os.path.isfile(config_file):
            print("running  add config for", images_dir)
            mean_img = utils_ballrot.get_mean_image(
                fictrac_video_file, skip_existing=True
            )
            x, y, r = utils_ballrot.get_ball_parameters(
                mean_img, output_dir=images_dir
            )
            roi_circ = utils_ballrot.get_circ_points_for_config(
                x, y, r, mean_img.shape
            )
            config_file = utils_ballrot.write_config_file(
                fictrac_video_file, roi_circ, overwrite=False
            )
        if not os.path.isfile(
            os.path.join(images_dir, f"camera_{fictrac_cam}-configImg.png")
        ):
            utils_ballrot.run_fictrac_config_gui(config_file)
        if len(glob.glob(os.path.join(images_dir, f"camera_{fictrac_cam}-*.dat"))) == 0:
            print("running fictrac for", images_dir)
            subprocess.run(
                ["/home/aymanns/fictrac/bin/fictrac", "config.txt"],
                cwd=images_dir,
                stdout=subprocess.DEVNULL,
            )

for trial_dir in utils.load_exp_dirs("../../../recordings.txt")
    images_dir = os.path.join(trial_dir, "behData/images")
    fictrac_processing(images_dir)
