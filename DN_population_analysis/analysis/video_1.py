import os.path
import utils_video
import utils_video.generators
import utils2p
import utils2p.synchronization
import numpy as np

import DN_population_analysis.utils as utils


for trial_dir in utils.load_exp_dirs("../trials_for_paper_overall_ref.txt"):
    fly_dir = utils.get_fly_dir(trial_dir)
    date = utils.get_date(trial_dir)
    trial = utils.get_trial_number(trial_dir)
    if date != 210830 or trial != 3:
        continue

    video_file = os.path.join(trial_dir, "behData/images/camera_5.mp4")
    path_red_stack = os.path.join(trial_dir, "2p/red.tif")
    path_green_stack = os.path.join(trial_dir, "2p/green.tif")
    path_dff_stack = os.path.join(trial_dir, "2p/dff.tif")
    crop_param_file = os.path.join(fly_dir, "crop_parameters.csv")

    fly_df = utils.load_fly_data([trial_dir,], behaviour=True, fictrac=True, trajectory=True, joint_positions=True, odor=True, dFF=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["ACV", "MSC", "H2O"]), :]
    fly_df["Odor"] = fly_df["Odor"].map({"H2O": u"H\u2082O", "ACV": u"ACV", "MSC": u"MSC"})
    fly_df = utils.neural_dim_reduction(fly_df, 3, algo="lda")
    fly_df["Dim"] = fly_df["Dim"].map(str)
    index_cols = [c for c in fly_df.columns if c != "Dim" and c != "weight"]
    fly_df = fly_df.pivot(index=index_cols, columns="Dim", values="weight").reset_index()
    fly_df = fly_df.sort_values("Frame")
    
    sync_file = utils2p.find_sync_file(trial_dir)
    metadata_file = utils2p.find_metadata_file(trial_dir)
    metadata = utils2p.Metadata(metadata_file)
    sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
    seven_camera_metadata_file = utils2p.find_seven_camera_metadata_file(trial_dir)
    frame_counter, cam_line = utils2p.synchronization.get_lines_from_h5_file(sync_file, ("Frame Counter", "Cameras"))
    frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata=metadata)
    cam_line = utils2p.synchronization.process_cam_line(cam_line, seven_camera_metadata_file)

    # Correct for denoising kernel
    frame_counter = frame_counter - 30

    crop_mask = (frame_counter > 0) & (cam_line > 0)
    frame_counter, cam_line = utils2p.synchronization.crop_lines(crop_mask, (frame_counter, cam_line))

    #start = max(fly_df["Frame"].min(), cam_line[0])
    #stop = fly_df["Frame"].max()
    start = 26200
    stop = 36600

    fly_df = fly_df.loc[(fly_df["Frame"] >= start) & (fly_df["Frame"] < stop), :]

    beh_cam_generator = utils_video.generators.video(video_file,
                                                     start=start,
                                                     stop=stop,
                                                    )
    odor_text = fly_df["Odor"].values
    odor_color = [utils.behaviour_colours[odor] for odor in odor_text]
    beh_cam_generator = utils_video.generators.add_text_PIL(beh_cam_generator, odor_text, color=odor_color, size=60, pos=(780, 80))

    red_stack = utils2p.load_img(path_red_stack, memmap=True)[30:-30]
    green_stack = utils2p.load_img(path_green_stack, memmap=True)[30:-30]
    red_stack, com_offsets = utils.center_stack(red_stack, return_offset=True)
    green_stack = utils.apply_offset(green_stack, com_offsets)
    height_offset, height, width_offset, width = utils.read_crop_parameters(crop_param_file)
    height_offset = height_offset - com_offsets[0, 0]
    width_offset = width_offset - com_offsets[1, 0]
    red_stack = red_stack[:, height_offset : height_offset + height, width_offset : width_offset + width]
    green_stack = green_stack[:, height_offset : height_offset + height, width_offset : width_offset + width]

    raw_generator = utils_video.generators.frames_2p(red_stack, green_stack, percentiles=(2, 99.5), size=(480, -1))
    raw_generator = utils_video.generators.add_text_PIL(raw_generator, "GCaMP6s", size=60, pos=(10, 340), color="#0FF")
    raw_generator = utils_video.generators.add_text_PIL(raw_generator, "tdTomato", size=60, pos=(10, 410), color="#F00")

    raw_generator_shape, raw_generator = utils_video.utils.get_generator_shape(raw_generator)

    dff_stack = utils2p.load_img(path_dff_stack, memmap=True)
    dff_generator = utils_video.generators.dff(dff_stack, size=(raw_generator_shape[0], -1), cbar_pos="right", font_size=32, cbar_size=200, cmap=utils.cmap)
    
    dff_generator_shape, dff_generator = utils_video.utils.get_generator_shape(dff_generator)

    raw_generator = utils_video.generators.pad(raw_generator, 0, 0, 0, dff_generator_shape[1] - raw_generator_shape[1])
    raw_generator_shape, raw_generator = utils_video.utils.get_generator_shape(raw_generator)


    if (fly_df["trajectory_y"].max() - fly_df["trajectory_y"].min()) > (fly_df["trajectory_x"].max() - fly_df["trajectory_x"].min()):
        tmp = fly_df["trajectory_x"].copy()
        fly_df["trajectory_x"] = -fly_df["trajectory_y"]
        fly_df["trajectory_y"] = tmp
        fly_df["heading"] = fly_df["heading"] + np.pi / 2
    fly_df["trajectory_color"] = "gray"
    fly_df.loc[fly_df["Frame"] > 29300, "trajectory_color"] = "white"
    trajectory_generator = utils_video.generators.trajectory(fly_df["trajectory_x"].values,
                                                             fly_df["trajectory_y"].values,
                                                             size=(dff_generator_shape[0], -1),
                                                             color="white",
                                                             color_window=200,
                                                             past_color="gray",
                                                            )

    beh_cam_generator_shape, beh_cam_generator = utils_video.utils.get_generator_shape(beh_cam_generator)
    trajectory_generator_shape, trajectory_generator = utils_video.utils.get_generator_shape(trajectory_generator)
    if trajectory_generator_shape[1] > beh_cam_generator_shape[1]:
        beh_cam_generator = utils_video.generators.pad(beh_cam_generator, 0, 0, trajectory_generator_shape[1] - beh_cam_generator_shape[1], 0)
    else:
        padding_width = beh_cam_generator_shape[1] - trajectory_generator_shape[1]
        trajectory_generator = utils_video.generators.pad(trajectory_generator, 0, 0, int(np.ceil(padding_width / 2)), int(np.floor(padding_width / 2)))
    
    beh_cam_generator_shape, beh_cam_generator = utils_video.utils.get_generator_shape(beh_cam_generator)
    trajectory_generator_shape, trajectory_generator = utils_video.utils.get_generator_shape(trajectory_generator)

    df3d_generator = utils_video.generators.df3d_line_plots_df(fly_df, groupby=["Trial",], dim="3d", size=(-1, beh_cam_generator_shape[1]))
    df3d_generator_shape, df3d_generator = utils_video.utils.get_generator_shape(df3d_generator)

    beh_text = [utils.rename_behaviour(beh) for beh in fly_df["Behaviour"].values]
    beh_color = [utils.behaviour_colours[beh] for beh in beh_text]
    beh_text = ["\n".join(t.rsplit(" ", 1)) for t in beh_text]
    beh_text = [t if t != "Background" else "" for t in beh_text]
    df3d_generator = utils_video.generators.add_text_PIL(df3d_generator, beh_text, color=beh_color, size=60, pos=(680, 0))

    beh_generators = [beh_cam_generator, trajectory_generator, df3d_generator]
    beh_generator = utils_video.generators.stack(beh_generators, axis=0)

    beh_generator_shape, beh_generator = utils_video.utils.get_generator_shape(beh_generator)


    pcs = fly_df[["0", "1", "2"]].values
    pca_generator = utils_video.generators.dynamics_3D(pcs, n=0, size=df3d_generator_shape[:2])#, font_size=20, xticks=[-2, 0, 2], yticks=[-2, 0, 2], zticks=[-2, 0, 2])
    pca_generator_shape, pca_generator = utils_video.utils.get_generator_shape(pca_generator)
    pca_generator = utils_video.generators.pad(pca_generator, 0, 0, 0, dff_generator_shape[1] - pca_generator_shape[1])
    pca_generator_shape, pca_generator = utils_video.utils.get_generator_shape(pca_generator)

    neural_generator = utils_video.generators.stack([raw_generator, dff_generator])

    neural_indices = utils2p.synchronization.beh_idx_to_2p_idx(np.arange(start, stop), cam_line, frame_counter)
    neural_generator =  utils_video.generators.resample(neural_generator, neural_indices)
    
    neural_generator = utils_video.generators.stack([neural_generator, pca_generator])
    
    generator = utils_video.generators.stack([beh_generator, neural_generator], axis=1)

    generator = utils_video.generators.pad(generator, 30, 30, 30, 30)
    
    utils_video.make_video(f"summary_{date}_{trial:03}_cropped.mp4", generator, 100, output_shape=(1800, -1))#, n_frames=1000)
