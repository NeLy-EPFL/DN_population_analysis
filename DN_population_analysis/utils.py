import pathlib
import os.path
import pickle
import collections
import csv

import pandas as pd
import numpy as np
import scipy
import sklearn.model_selection
import sklearn.neighbors
import glmnet_python
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import skimage
import utils2p
import utils2p.synchronization

import utils_ballrot


PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]

BEHAVIOUR_COLOURS = {
        "Walking": '#E05361', "Resting": '#B7BA78', "Head grooming": '#8A4FF7', "Front leg rubbing": '#4CC8ED', "Posterior movements": '#F7B14F', "Undefined": "#FFFFFF", "Posterior grooming": '#F7B14F',
        "walking": '#E05361', "resting": '#B7BA78', "head_grooming": '#8A4FF7', "foreleg_grooming": '#4CC8ED', "hind_grooming": '#F7B14F', 
        "ACV": "#FF1493", "MSC": "#00FF00", "H2O": '#6a6a6a', "H\u2082O": '#6a6a6a',
        "undefined": "#FFFFFF",
        "background": "#FFFFFF",
        "Background": "#FFFFFF",
        "": "#FFFFFF",
        }

BEHAVIOUR_LINEAR = {
        "Walking": 0, "Resting": 0.2, "Head grooming": 0.4, "Front leg rubbing": 0.6, "Hind grooming": 0.8, "Undefined": 1,
        "walking": 0, "resting": 0.2, "head_grooming": 0.4, "foreleg_grooming": 0.6, "hind_grooming": 0.8, "undefined": 1
        }

BEHAVIOUR_CLASSES = ["walking", "resting", "head_grooming", "frontleg_grooming", "hind_grooming"]

SEABORN_RC={"figure.figsize": (3, 3),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.major.pad": 0.75,
    "ytick.major.pad": 0.75,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "font.family": "serif",
    "font.serif":["Arial"],
    "lines.linewidth": 0.5,
    "axes.labelsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "xtick.bottom": True,
    "ytick.left": True,
    }

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.linewidth"] = 0.5
matplotlib.rcParams["xtick.major.width"] = 0.5
matplotlib.rcParams["ytick.major.width"] = 0.5
matplotlib.rcParams["xtick.major.size"] = 1.5
matplotlib.rcParams["ytick.major.size"] = 1.5
matplotlib.rcParams["xtick.major.pad"] = 0.75
matplotlib.rcParams["ytick.major.pad"] = 0.75
matplotlib.rcParams["xtick.minor.width"] = 0.3
matplotlib.rcParams["ytick.minor.width"] = 0.3
matplotlib.rcParams["xtick.minor.size"] = 1
matplotlib.rcParams["ytick.minor.size"] = 1
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 6
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.it"] = "Arial"
matplotlib.rcParams["mathtext.rm"] = "Arial"
matplotlib.rc("font", **{"family":"serif", "serif":["Arial"]})
matplotlib.rc("lines", linewidth=0.5)
matplotlib.rc("axes", labelsize=6)

def load_exp_dirs(path):
    dirs = []
    with open(path, "r") as f:
        for line in f:
            if line[:1] == "#":
                continue
            line = line.rstrip()
            directory = os.path.join(PACKAGE_DIR, line)
            dirs.append(directory)
    return dirs

def get_fly_dir(path):
    path = path.rstrip("/")
    while not path.split("/")[-1].startswith("Fly"):
        path = os.path.dirname(path)
    return path

def get_date(path):
    # This is a hack to keep the data path on the Dataverse simple normally the date should be contained in the path
    dates = {1: 210830, 2: 210910, 3: 211026, 4: 211027, 5: 211029}
    return dates[get_fly_number(path)]

def group_by_fly(dirs):
    groups = {}
    fly_dirs = list(map(get_fly_dir, dirs))
    for fly_dir in sorted(set(fly_dirs)):
        groups[fly_dir] = [d for d in dirs if fly_dir in d]
    return groups

def get_trial_number(path):
    trial_number = int(path.split("_trial")[1][-3:])
    return trial_number

def get_fly_number(path):
    path = get_fly_dir(path)
    fly_number = int(path.split("Fly")[1])
    return fly_number

def get_genotype(path):
    fly_dir = get_fly_dir(path)
    genotype = fly_dir.rstrip("/").split("/")[-2]
    return genotype

def load_R65D11_data(trial_dirs, moving_average=35):
    df = pd.DataFrame()
    for trial_dir in trial_dirs:
        trial = get_trial_number(trial_dir)

        # Behavior
        behaviour_file = os.path.join(trial_dir, "behData/images/df3d/behaviour_predictions_daart.pkl")
        with open(behaviour_file, "rb") as fh:
            beh_df = pickle.load(fh)
        beh_df = beh_df.rename(columns={"Prediction": "Behaviour"})
        beh_df = beh_df.reset_index()

        # hack to match the trial naming convention of the data uploaded to the dataverse
        beh_df["Trial"] = beh_df["Trial"] - 2

        # CO2
        h5_path = utils2p.find_sync_file(trial_dir)
        co2_line, cam_line, frame_counter, capture_on = utils2p.synchronization.get_lines_from_h5_file(h5_path, ["CO2_Stim", "Basler", "Frame Counter", "Capture On"])
        try:
            capture_json = utils2p.find_seven_camera_metadata_file(trial_dir)
        except FileNotFoundError:
            capture_json = None
        metadata_2p = utils2p.find_metadata_file(trial_dir)
        metadata = utils2p.Metadata(metadata_2p)
        cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
        n_flyback_frames = metadata.get_n_flyback_frames()
        n_steps = metadata.get_n_z()
        frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, steps_per_frame=n_flyback_frames + n_steps)
        co2_line = utils2p.synchronization.process_stimulus_line(co2_line)
    
        mask = np.logical_and(capture_on, frame_counter >= 0)
        mask = np.logical_and(mask, cam_line >= 0)
        co2_line, cam_line, frame_counter = utils2p.synchronization.crop_lines(mask, [co2_line, cam_line, frame_counter])

        sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)
        sync_metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
        freq = sync_metadata.get_freq()
        times = utils2p.synchronization.get_times(len(cam_line), freq)
        frame_times = utils2p.synchronization.get_start_times(cam_line, times, zero_based_counter=True)
        frame_times_2p = utils2p.synchronization.get_start_times(frame_counter, times, zero_based_counter=True)

        co2 = utils2p.synchronization.reduce_during_frame(cam_line, co2_line, np.mean)

        beh_df["CO2"] = co2

        # 2P traces
        dff_file = os.path.join(trial_dir, "2p/output/GC6_auto/final/DFF_dic.p")
        with open(dff_file, "rb") as f:
            dff = pickle.load(f)
        neural_df = pd.DataFrame()
        for roi, key in enumerate(sorted(dff.keys())):
            dff[key] = np.array(dff[key])
            dff[key] = interpolate_for_nans(dff[key])
            dff[key] = dff[key][np.min(frame_counter) : np.max(frame_counter) + 1]
            interpolator = scipy.interpolate.interp1d(frame_times_2p, dff[key], kind="linear", bounds_error=False, fill_value="extrapolate")
            dff[key] = interpolator(frame_times)
            if moving_average != 0:
                kernel = np.ones(moving_average) / moving_average
                dff[key] = np.convolve(dff[key], kernel, "same")
            roi_df = pd.DataFrame()
            roi_df["dFF"] = dff[key]
            roi_df["ROI"] = roi
            roi_df["Date"] = 181221
            roi_df["Genotype"] = "R65D11-tdTomGC6fopt"
            roi_df["Fly"] = 1
            roi_df["Trial"] = trial
            roi_df["Frame"] = np.arange(np.min(cam_line) , np.max(cam_line) + 1)
            neural_df = pd.concat([neural_df, roi_df], axis=0)
        
        trial_df = pd.merge_ordered(neural_df,
                                    beh_df,
                                    left_by="ROI",
                                    left_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
                                    right_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
                                    how="left"
                                   )

        df = pd.concat([df, trial_df], axis=0)
    return df

def interpolate_for_nans(traces):
    """
    This function linearly interpolates nan values in the traces.

    Parameters
    ----------
    traces : numpy array
        Trace of values that contain nans.
        Second dimension is time. If 1D, first
        dimension is time.
    """
    if traces.ndim > 2 or traces.ndim == 0:
        raise ValueError("Parameter 'traces' must be a 1D or 2D numpy array.")
    if traces.ndim == 1:
        traces = np.expand_dims(traces, axis=0)

    mask_nan = np.isnan(traces)
    x = np.arange(traces.shape[1])
    for i in range(traces.shape[0]):
        if np.sum(mask_nan[i]) == 0 or np.sum(~mask_nan[i]) == 0:
            continue
        interp_locations = x[mask_nan[i]]
        value_locations = x[~mask_nan[i]]
        values = traces[i, ~mask_nan[i]]
        traces[i, mask_nan[i]] = np.interp(interp_locations, value_locations, values)
    return np.squeeze(traces)

def behaviour_shading(axis, x, behaviours):
    n_beh = len(np.unique(behaviours))
    for i, beh in enumerate(np.unique(behaviours)):
        if beh == "background" or beh == "":
            continue
        stimulus_shading(axis, x, behaviours == beh, color=BEHAVIOUR_COLOURS[beh], alpha=1)

def stimulus_shading(axis, x, stimulus, color="black", alpha=0.5):
    binary = np.zeros(len(stimulus) + 2, dtype=bool)
    binary[1:-1] = stimulus
    start_indices = utils2p.synchronization.edges(binary, size=(0, np.inf))[0] - 1
    stop_indices = utils2p.synchronization.edges(binary, size=(-np.inf, 0))[0] - 2
    for start, stop in zip(start_indices, stop_indices):
        axis.axvspan(x[start], x[stop], alpha=alpha, color=color, linewidth=0)

def load_fly_data(trial_dirs, dFF=False, behaviour=False, fictrac=False, angles=False, active=False, odor=False, joint_positions=False, trajectory=False, frames_2p=False, antenna_tarsus_dist=False, antenna_touches=False, derivatives=False):
    fly_df = pd.DataFrame()
    for trial_dir in trial_dirs:
        print("\t" + trial_dir)
        date = get_date(trial_dir)
        genotype = get_genotype(trial_dir)
        fly = get_fly_number(trial_dir)
        trial = get_trial_number(trial_dir)

        behaviour_file = os.path.join(trial_dir, "behData/images/df3d/behaviour_predictions_daart.pkl")
        fictrac_file = utils2p.find_fictrac_file(trial_dir, most_recent=True)
        pose_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
        touches_file = os.path.join(trial_dir, "behData/images/df3d/antenna_touches_new_sdf.pkl")
        roi_dff_file = os.path.join(trial_dir, "2p/roi_dFF.pkl")
        active_file = os.path.join(trial_dir, "2p/roi_binary.pkl")
        derivative_file = os.path.join(trial_dir, "2p/derivatives_5000.pkl")

        if behaviour:
            with open(behaviour_file, "rb") as fh:
                beh_df = pickle.load(fh)
            beh_df = beh_df.rename(columns={"Prediction": "Behaviour"})
            beh_df = beh_df.reset_index()
            beh_df["Genotype"] = genotype
            beh_df["Fly"] = fly

        if fictrac:
            fictrac_df = load_fictrac(fictrac_file, date, genotype, fly, trial, trajectory=trajectory)
            if behaviour:
                beh_df = beh_df.merge(fictrac_df, on=("Date", "Genotype", "Fly", "Trial", "Frame"))
            else:
                beh_df = fictrac_df

        if angles:
            with open(pose_file, "rb") as fh:
                joint_angles = pickle.load(fh)
            joint_angles = joint_angles.filter(like="Angle")
            joint_angles = joint_angles.reset_index()
            joint_angles["Genotype"] = genotype
            joint_angles["Fly"] = fly
            if not behaviour and not fictrac:
                beh_df = joint_angles
            else:
                beh_df = beh_df.merge(joint_angles, on=("Date", "Genotype", "Fly", "Trial", "Frame"))

        if odor:
            odor_df = load_odor_data(trial_dir)
            if not behaviour and not fictrac and not angles:
                beh_df = odor_df
            else:
                beh_df = beh_df.merge(odor_df, on=("Date", "Genotype", "Fly", "Trial", "Frame"))

        if joint_positions:
            if not isinstance(joint_positions, str):
                joint_positions = "Pose"
            with open(pose_file, "rb") as fh:
                joint_position_df = pickle.load(fh)
            joint_position_df = joint_position_df.filter(like=joint_positions)
            joint_position_df = joint_position_df.reset_index()
            joint_position_df["Genotype"] = genotype
            joint_position_df["Fly"] = fly
            if not behaviour and not fictrac and not angles and not odor:
                beh_df = joint_position_df
            else:
                beh_df = beh_df.merge(joint_position_df, on=("Date", "Genotype", "Fly", "Trial", "Frame"))
        if frames_2p:
            frame_indices = load_2p_frame_indices(trial_dir)
            if not behaviour and not fictrac and not angles and not odor and not joint_positions:
                beh_df = frame_indices
            else:
                beh_df = beh_df.merge(frame_indices, on=("Date", "Genotype", "Fly", "Trial", "Frame"))

        if antenna_tarsus_dist:
            dist_df = load_antenna_tarsus_distance(pose_file, date, genotype, fly, trial)
            if not behaviour and not fictrac and not angles and not odor and not joint_positions and not frames_2p:
                beh_df = dist_df
            else:
                beh_df = beh_df.merge(dist_df, on=("Date", "Genotype", "Fly", "Trial", "Frame"))
        
        if antenna_touches:
            touches_df = load_antenna_touches(touches_file, date, genotype, fly, trial)
            if not behaviour and not fictrac and not angles and not odor and not joint_positions and not frames_2p and not antenna_tarsus_dist:
                beh_df = touches_df
            else:
                beh_df = beh_df.merge(touches_df, on=("Date", "Genotype", "Fly", "Trial", "Frame"))
        
        if dFF:
            with open(roi_dff_file, "rb") as fh:
                neural = pickle.load(fh)
            neural = neural.loc[neural["Source file"] == "denoised", :]
        if active:
            with open(active_file, "rb") as fh:
                active_df = pickle.load(fh)
            event_based_index, event_number = utils2p.synchronization.event_based_frame_indices(active_df["active"])
            active_df["event_number"] = event_number
            active_df["event_based_index"] = event_based_index
            if not dFF:
                neural = active_df
            else:
                neural = neural.merge(active_df, on=("Date", "Genotype", "Fly", "Trial", "Frame", "ROI"))
        if derivatives:
            with open(derivative_file, "rb") as fh:
                derivative_df = pickle.load(fh)
            derivative_df = upsample_df_to_beh(trial_dir, derivative_df)
            if not dFF and not active:
                neural = derivative_df
            else:
                neural = neural.merge(derivative_df, on=("Date", "Genotype", "Fly", "Trial", "Frame", "ROI"))
        if not dFF and not active and not derivatives:
            fly_df = fly_df.append(beh_df)
            continue
        
        neural = neural.reset_index()
        neural["Genotype"] = genotype
        neural["Fly"] = fly

        trial_df = pd.merge_ordered(neural,
                                    beh_df,
                                    left_by="ROI",
                                    left_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
                                    right_on=("Date", "Genotype", "Fly", "Trial", "Frame"),
                                    how="left"
                                   )

        fly_df = pd.concat([fly_df, trial_df], axis="index")
    return fly_df

def load_odor_data(trial_dir):
    date = get_date(trial_dir)
    genotype = get_genotype(trial_dir)
    fly = get_fly_number(trial_dir)
    trial = get_trial_number(trial_dir)
    if "ref" in trial_dir:
        sync_file = utils2p.find_sync_file(trial_dir[:-9])
        capture_json = utils2p.find_seven_camera_metadata_file(trial_dir[:-9])
        metadata_file = utils2p.find_metadata_file(trial_dir[:-9])
    else:
        sync_file = utils2p.find_sync_file(trial_dir)
        capture_json = utils2p.find_seven_camera_metadata_file(trial_dir)
        metadata_file = utils2p.find_metadata_file(trial_dir)
    cam_line, odor_line = utils2p.synchronization.get_lines_from_h5_file(sync_file, ("Cameras", "odor"))
    cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)
    arduino_commands = ("",) * 14 + ("H2O", "ACV", "MSC", "")
    odor_line = utils2p.synchronization.process_odor_line(odor_line, step_size=0.26965, arduino_commands=arduino_commands)
    mask = cam_line >= 0
    cam_line, odor_line = utils2p.synchronization.crop_lines(mask, (cam_line, odor_line))
    majority_func = lambda x: collections.Counter(x).most_common(1)[0][0]
    odor_beh_frames = utils2p.synchronization.reduce_during_frame(cam_line, odor_line, majority_func)
    frames = np.arange(len(odor_beh_frames))
    n_frames = len(frames)
    odor_index = pd.MultiIndex.from_arrays(
            [
                np.array([date,]  * n_frames),
                np.array([genotype,] * n_frames),
                np.array([fly,] * n_frames),
                np.array([trial,] * n_frames),
                frames,
            ],
            names=("Date", "Genotype", "Fly", "Trial", "Frame")
        )
    odor_df = pd.DataFrame(index=odor_index)
    odor_df["Odor"] = odor_beh_frames
    return odor_df

def convolve_with_crf(t, x, trials, a=7.4, b=0.3):
    t = t - np.min(t)
    t = t[t < 30]
    # In convolution mode same the shorter sequence is used as kernel
    # Ensures crf is the shorter sequence
    t = t[:int(len(t) / 2) - 1]
    crf = -np.exp(-a * t) + np.exp(-b * t)
    crf = crf / sum(crf)
    crf = np.concatenate((np.zeros(len(crf) - 1), crf))

    if trials is None:
        trials = np.zeros(len(x), dtype=int)

    conv_x = np.zeros(x.shape, dtype=float)
    for trial in np.unique(trials):
        trial_mask = trials == trial
        trial_x = x[trial_mask]
        trial_x = np.pad(trial_x, len(t), mode="edge")
        conv_x[trial_mask] =  np.convolve(trial_x, crf, mode="same")[len(t):-len(t)]
    return conv_x

def cv_rsquared(date, fly, var, roi, exog, endog, reduced_exog, trials, bounds, reduced_bounds, one_model_per_trial=False, prediction_plots_dir=None):
    results = pd.DataFrame()

    if one_model_per_trial:
        for trial in np.unique(trials):
            trial_mask = trials == trial

            trial_exog = exog[trial_mask].copy()
            non_const_regressors = np.std(trial_exog, axis=0) > 1e-10
            trial_exog = trial_exog[:, non_const_regressors].copy()
            trial_bounds = bounds[:, non_const_regressors].copy() if bounds is not None else None

            trial_reduced_exog = reduced_exog[trial_mask].copy()
            non_const_regressors = np.std(trial_reduced_exog, axis=0) > 1e-10
            trial_reduced_exog = trial_reduced_exog[:, non_const_regressors].copy()
            trial_reduced_bounds = reduced_bounds[:, non_const_regressors].copy() if reduced_bounds is not None else None
            
            trial_results = cv_rsquared(date, fly, var, roi, trial_exog,
                    endog[trial_mask].copy(), trial_reduced_exog, trials[trial_mask].copy(),
                    trial_bounds, trial_reduced_bounds, one_model_per_trial=False, prediction_plots_dir=prediction_plots_dir)
            trial_results["Trial"] = trial
            results = pd.concat([results, trial_results], axis="index")
        return results

    ground_truth = []
    full_prediction = []
    reduced_prediction = []

    skf = sklearn.model_selection.StratifiedKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(skf.split(exog, trials)):
        train_exog = exog[train_index]
        train_reduced_exog = reduced_exog[train_index]
        train_endog = endog[train_index]
        test_exog = exog[test_index]
        test_reduced_exog = reduced_exog[test_index]
        test_endog = endog[test_index]

        if len(np.unique(train_endog)) == 1 or len(np.unique(test_endog)) == 1:
            print("Either the train or test endogenous variable is constant")
            continue

        train_trials = trials[train_index]

        prediction = predict(train_exog, train_endog, test_exog, bounds=bounds, train_trials=train_trials)
        full_model_rsquared = rsquared(test_endog, prediction)
        full_rmse = rmse(test_endog, prediction)
        full_sse = np.sum((test_endog - prediction) ** 2)

        ground_truth.append(test_endog)
        full_prediction.append(np.squeeze(prediction))

        prediction = predict(train_reduced_exog, train_endog, test_reduced_exog, bounds=reduced_bounds, train_trials=train_trials)
        reduced_model_rsquared = rsquared(test_endog, prediction)
        reduced_rmse = rmse(test_endog, prediction)
        reduced_sse = np.sum((test_endog - prediction) ** 2)

        reduced_prediction.append(np.squeeze(prediction))

        current_rsquared = max(0, max(0, full_model_rsquared) - max(0, reduced_model_rsquared))
        print(current_rsquared, full_model_rsquared, reduced_model_rsquared)
        current_weight = 0

        row_df = pd.DataFrame({"Fly": (f"{date}_{fly}",),
                               "Variable": var,
                               "ROI": (roi,),
                               "Fold": (fold,),
                               "rsquared": (current_rsquared,),
                               "full_model_rsquared": (full_model_rsquared,),
                               "reduced_model_rsquared": (reduced_model_rsquared,),
                               "weight": (current_weight,),
                               "full_rmse": full_rmse,
                               "reduced_rmse": reduced_rmse,
                               "full_sse": full_sse,
                               "reduced_sse": reduced_sse,
                               "len_train_endog": len(train_endog),
                               "len_test_endog": len(test_endog),
                               "full_n_regressors": exog.shape[1],
                               "reduced_n_regressors": reduced_exog.shape[1],
                              }
                             )

        results = pd.concat([results, row_df], axis="index")

    if prediction_plots_dir is not None and len(full_prediction) > 0 and len(reduced_prediction) > 0:
        full_prediction = np.concatenate(full_prediction)
        reduced_prediction = np.concatenate(reduced_prediction)
        ground_truth = np.concatenate(ground_truth)
        
        total_rsquared = rsquared(ground_truth, full_prediction)
        reduced_rsquared = rsquared(ground_truth, reduced_prediction)
        unique_rsquared = max(0, max(0, total_rsquared) - max(0, reduced_rsquared))
        if total_rsquared > 0.05: 
            t = np.arange(len(ground_truth)) / 100
            if len(np.unique(trials)) > 1:
                trial=0
            else:
                trial = trials[0]
            output_path = os.path.join(prediction_plots_dir, f"{var}_{date}_{trial:03}_ROI_{int(roi)}.pdf")
            ylabel = "%dF/F"
            annotation = f"$R^2$={total_rsquared:.2f}\n$R^2$={reduced_rsquared:.2f}\nUEV={unique_rsquared:.2f}"
            plot_prediction(t, reduced_prediction, ground_truth, output_path, ylabel, annotation=annotation, prediction2=full_prediction)
    return results

def predict(train_exog, train_endog, test_exog, bounds=None, train_trials=None):
    # if there are no regressors, i.e. only constant regressors the mean is returned
    if train_exog.shape[1] == 0 or (tuple(np.unique(train_exog)) == (0,)):
        return np.ones(test_exog.shape[0]) * np.mean(train_endog)
    try:
        fit = get_fit(train_exog, train_endog, bounds=bounds, trials=train_trials)
        prediction = cvglmnetPredict(fit, test_exog, s="lambda_min")[:, 0]
    except ValueError as e:
        print(e)
        return np.ones(len(test_exog)) * np.nan
    return prediction

def get_fit(exog, endog, bounds=None, trials=None, n_folds=5):
    if bounds is None:
        bounds = np.zeros((2, exog.shape[1]))
        bounds[0, :] = np.NINF
        bounds[1, :] = np.inf

    if trials is None:
        trials = np.zeros(exog.shape[0], dtype=int)

    foldid = np.ones(exog.shape[0], dtype=int)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds)
    for fold, (index, test_index) in enumerate(skf.split(exog, trials)):
        foldid[test_index] = fold
    fit = cvglmnet(x=exog, y=endog, alpha=0, cl=bounds, intr=True, ptype="mse", foldid=foldid, parallel=True)
    return fit

def rsquared(y, yhat):
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - sse / sst

def rmse(y, yhat):
    return np.sqrt(np.mean((yhat - y) ** 2))

def plot_prediction(t, prediction, ground_truth, output_path, ylabel, annotation=None, prediction2=None):
    #fig = plt.figure(figsize=(3, 0.5))
    fig = plt.figure(figsize=(3, 1))
    plt.xlim((np.min(t) - 10, np.max(t) + 10))
    plt.plot(t, ground_truth, color="black", linewidth=0.5)
    plt.plot(t, prediction, color="cyan", linewidth=0.5)
    if prediction2 is not None:
        plt.plot(t, prediction2, color="red", linewidth=0.5)
    if annotation is not None:
        plt.annotate(annotation, (np.max(t) * 0.8, min(np.min(prediction), np.min(ground_truth))), fontsize=6)
    plt.xlabel("Time [s]", fontsize=6)
    plt.ylabel(ylabel, fontsize=6)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()

def rename_behaviour(beh):
    rename_dict = {"walking": "Walking", "head_grooming": "Head grooming", "resting": "Resting", "hind_grooming": "Posterior movements", "foreleg_grooming": "Front leg rubbing", "ACV": "ACV", "MSC": "MSC", "all_odors": "All odors", "all_resting": "Resting", "turn_r": "Right turning", "turn_l": "Left turning", "turn": "Turning", "noair": "No air", "background": "Background"}
    return rename_dict[beh]

def remove_axis(ax):
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

def pick_max_behaviour(df):
    df = df.drop(columns=["ROI", "Fly"])
    return df.loc[df["rsquared"].idxmax(), :]

def generate_position_plots(trial_dirs, out_dir, out_name, results_df, color_var, size_var, rsquared_thresh=0.05, cmap=matplotlib.cm.viridis, reduction_function=None, numbers=False, vmax_size=None, figsize=(3, 0.8), percent=True, kde=False):

    if not "Variable" in results_df.columns:
        results_df = results_df.groupby(["Fly", "ROI"]).mean().reset_index()
    else:
        results_df = results_df.groupby(["Fly", "ROI", "Variable"]).mean().reset_index()

    if reduction_function is not None:
        results_df = results_df.groupby(["Fly", "ROI"]).apply(reduction_function).reset_index()

    #results_df["rsquared"] = np.abs(results_df["rsquared"])

    if vmax_size is None:
        vmax_size = np.max(results_df[size_var])
    vmin_colour = np.min(results_df[color_var])
    vmax_colour = np.max(results_df[color_var])

    if "rsquared" in results_df.columns:
        results_df = results_df.loc[results_df["rsquared"] > rsquared_thresh, :]
        if results_df.shape[0] == 0:
            return

    for trial_dir in trial_dirs:
        date = get_date(trial_dir)
        fly = get_fly_number(trial_dir)
        trial = get_trial_number(trial_dir)

        fly_results = results_df.loc[results_df["Fly"] == f"{date}_{fly}", :]
        if fly_results.shape[0] == 0:
            continue
    
        output_file = os.path.join(out_dir, f"positions_{out_name}_{date}_Fly_{fly}_{trial:03}.pdf")
        roi_location_plot(trial_dir, 
                          fly_results["ROI"].values.astype(int),
                          fly_results[color_var].values,
                          fly_results[size_var].values,
                          output_file,
                          vmax_radius=vmax_size,
                          vmin_colour=vmin_colour,
                          vmax_colour=vmax_colour,
                          cmap=cmap,
                          numbers=numbers,
                          figsize=figsize,
                          percent=percent,
                          roi_labels=fly_results["New_ROI"].values if numbers else None,
                         )
        if kde:
            output_file = os.path.join(out_dir, f"kde_{out_name}_{date}_Fly_{fly}_{trial:03}_normalized.pdf")
            groups = []
            colors = []
            norm = matplotlib.colors.Normalize(vmin=vmin_colour, vmax=vmax_colour)
            roi_centers = np.loadtxt(os.path.join(trial_dir, "2p/roi_centers.txt"))
            for colour_val, df in fly_results.groupby(color_var):
                rois = df["ROI"].values.astype(int)
                rois_x = roi_centers[rois][:, 0]
                groups.append(rois_x)
                colors.append(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(colour_val))
            plot_kernel_density_estimate(output_file, groups, colors, x=np.arange(640), midline=320, bandwidth=[50,], all_data=roi_centers[:, 0])#np.linspace(1, 200, 50))

    if kde:
        output_file = os.path.join(out_dir, f"kde_{out_name}_normalized.pdf")
        groups = []
        colors = []
        norm = matplotlib.colors.Normalize(vmin=vmin_colour, vmax=vmax_colour)
        for colour_val, df in results_df.groupby(color_var):
            rois_x = []
            all_data = []
            for fly, fly_df in df.groupby("Fly"):
                date, fly_number = fly.split("_")
                date = int(date)
                trial_dir = sorted([d for d in trial_dirs if get_date(d) == date])[0]
                roi_centers = np.loadtxt(os.path.join(trial_dir, "2p/roi_centers.txt"))
                all_data.extend(list(roi_centers[:, 0]))
                rois = fly_df["ROI"].values.astype(int)
                rois_x.extend(list(roi_centers[rois][:, 0]))
            groups.append(np.array(rois_x))
            colors.append(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(colour_val))
        all_data = np.array(all_data)
        plot_kernel_density_estimate(output_file, groups, colors, x=np.arange(640), midline=320, bandwidth=[50,], all_data=all_data)#np.linspace(1, 200, 50))

def fix_image_size(stack, already_cropped=False, trial_dir=None, return_corrections=False):
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    
    target_height = 256
    target_width = 640

    if already_cropped:
        if stack.shape[2] < target_width:
            pad_size = int((target_width - stack.shape[2]) / 2)
            stack = np.pad(stack, pad_width=((0, 0), (0, 0), (pad_size, pad_size)), mode="constant", constant_values=1)
        elif stack.shape[2] > target_width:
            s = int((stack.shape[2] - target_width) / 2)
            stack = stack[:, :, s:-s]
        if stack.shape[1] < target_height:
            pad_size = int((target_height - stack.shape[1]) / 2)
            stack = np.pad(stack, pad_width=((0, 0), (pad_size, pad_size), (0, 0)), mode="constant", constant_values=1)
        elif stack.shape[1] > target_height:
            s = int((stack.shape[1] - target_height) / 2)
            stack = stack[:, s:-s, :]
        return np.squeeze(stack)

    height_offset, height, width_offset, width = read_crop_parameters(os.path.join(trial_dir, "2p/crop_parameters.csv"))
    if height != target_height:
        height_correction = int((height - target_height) / 2)
        height = target_height
        height_offset = height_offset + height_correction
    else:
        height_correction = 0
    if width != target_width:
        width_correction = int((width - target_width) / 2)
        width = target_width
        width_offset = width_offset + width_correction
    else:
        width_correction = 0
        
    cropped_stack = np.squeeze(stack[:, height_offset : height_offset + height, width_offset : width_offset + width])

    if return_corrections:
        return cropped_stack, (height_correction, width_correction)
    return cropped_stack

def read_crop_parameters(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        _ = next(reader)
        return tuple(map(int, next(reader)))

def draw_rois(trial_dir, ax, height_correction=0, width_correction=0, color="r"):
    roi_centers = np.loadtxt(os.path.join(trial_dir, "2p/roi_centers.txt"))
    roi_centers = roi_centers - np.array([height_correction, width_correction])
    patch_height, patch_width = 3, 2
    for roi_center in roi_centers:
        bottom_left = roi_center - np.array((patch_width, patch_height))
        patch = matplotlib.patches.Rectangle(bottom_left, width=4, height=6,linewidth=0.1, edgecolor=color, facecolor="none")
        ax.add_patch(patch)

def roi_location_plot(trial_dir, rois, colours, radii, output_file, vmax_radius=None, vmin_colour=None, vmax_colour=None, cmap=matplotlib.cm.viridis, numbers=False, figsize=(3, 0.8), percent=True, roi_labels=None):

    roi_centers = np.loadtxt(os.path.join(trial_dir, "2p/roi_centers.txt"))
    roi_centers = roi_centers[rois]
    
    if vmax_colour is None:
        vmax_colour = np.max(colours)
    if vmin_colour is None:
        vmin_colour = np.min(colours)
    
    mean_img = utils2p.load_img(os.path.join(trial_dir, "2p/mean_green.tif"))
    
    mean_img = fix_image_size(mean_img, trial_dir=trial_dir)

    mean_img = mean_img / np.max(mean_img)
    mean_img = skimage.exposure.equalize_adapthist(mean_img, clip_limit=0.03)
    
    _, cropped_height, _, cropped_width = read_crop_parameters(os.path.join(trial_dir, "2p/crop_parameters.csv"))
    new_img_shape = mean_img.shape
    height_correction = int((cropped_height - new_img_shape[0]) / 2)
    width_correction = int((cropped_width - new_img_shape[1]) / 2)
    
    mean_img = mean_img - np.percentile(mean_img, 5)
    mean_img = np.clip(mean_img, 0, None)
    mean_img = mean_img / np.max(mean_img)
    mean_img = np.array(mean_img * 255, dtype=np.uint8)
    
    mean_img = np.tile(mean_img[:, :, np.newaxis], (1, 1, 3))
    
    thickness = -1
   
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.imshow(mean_img)
    
    coordinates = []
    scatter_colours = []
    scatter_sizes = []
    radius_scaling_factor = 4
    for roi, roi_coordinates in zip(rois, roi_centers):
        coordinates.append(roi_coordinates - np.array((width_correction, height_correction)))
        roi_mask = rois == roi
        if sum(roi_mask) > 1:
            raise ValueError(f"There are {sum(roi_mask)} columns for ROI {roi}.")
        elif sum(roi_mask) == 0:
            print("empty roi mask")
            continue
        colour_val = colours[roi_mask][0]
        if radii is not None:
            radius = radii[roi_mask]
            radius = radius_scaling_factor * radius / vmax_radius
        else:
            radius = radius_scaling_factor
        scatter_sizes.append(radius)
        try:
            colour = BEHAVIOUR_COLOURS[colour_val]
            colour = tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))
            colour = tuple([c / 255 for c in colour])
        except KeyError:
            norm = matplotlib.colors.Normalize(vmin=vmin_colour, vmax=vmax_colour)
            colour = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(colour_val)
        scatter_colours.append(colour) 

    coordinates = np.array(coordinates)
    
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], s=scatter_sizes, c=scatter_colours)
    if numbers:
        if roi_labels is None:
            roi_labels = rois

        for roi, xy in zip(map(str, map(int, roi_labels)), coordinates):
            ax.annotate(roi, xy=xy, color="white", fontsize=3)

    if radii is not None:
        # Reduce size of image in plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if percent:
            inverse_size_transform = lambda s: s * vmax_radius / radius_scaling_factor * 100
            size_legend_elements = scatter.legend_elements(prop="sizes", num=5, color="gray", fmt="{x:0.0f} %", func=inverse_size_transform)
        else:
            inverse_size_transform = lambda s: s * vmax_radius / radius_scaling_factor
            size_legend_elements = scatter.legend_elements(prop="sizes", num=5, color="gray", fmt="{x:0.1f}", func=inverse_size_transform)
        legend_size = plt.legend(*size_legend_elements, title="R\u00B2", loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(size_legend_elements[0]), fontsize=4, handletextpad=0.1, frameon=False, labelspacing=0.3)
    plt.axis("off")
    plt.savefig(output_file, bbox_inches="tight", dpi=1200) 
    plt.close()

def upsample_df_to_beh(trial_dir, derivative_df):
    if "ref" in trial_dir:
        sync_file = utils2p.find_sync_file(trial_dir[:-9])
        capture_json = utils2p.find_seven_camera_metadata_file(trial_dir[:-9])
        metadata_file = utils2p.find_metadata_file(trial_dir[:-9])
        sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir[:-9])
    else:
        sync_file = utils2p.find_sync_file(trial_dir)
        capture_json = utils2p.find_seven_camera_metadata_file(trial_dir)
        metadata_file = utils2p.find_metadata_file(trial_dir)
        sync_metadata_file = utils2p.find_sync_metadata_file(trial_dir)

    cam_line, frame_counter = utils2p.synchronization.get_lines_from_h5_file(sync_file, ("Cameras", "Frame Counter"))
    metadata = utils2p.Metadata(metadata_file)
    frame_counter = utils2p.synchronization.process_frame_counter(frame_counter, metadata)
    cam_line = utils2p.synchronization.process_cam_line(cam_line, capture_json)

    mask = np.isin(frame_counter, derivative_df["Frame"])
    cam_line, frame_counter = utils2p.synchronization.crop_lines(mask, (cam_line, frame_counter))

    sync_metadata = utils2p.synchronization.SyncMetadata(sync_metadata_file)
    freq = sync_metadata.get_freq()
    times = utils2p.synchronization.get_times(len(cam_line), freq)
    cam_frame_indices = utils2p.synchronization.get_start_times(cam_line, cam_line, zero_based_counter=True)
    two_photon_frame_indices = utils2p.synchronization.get_start_times(frame_counter, frame_counter, zero_based_counter=True)
    frame_times = utils2p.synchronization.get_start_times(cam_line, times, zero_based_counter=True)
    frame_times_2p = utils2p.synchronization.get_start_times(frame_counter, times, zero_based_counter=True)

    upsampled_df = pd.DataFrame()
    for roi in derivative_df["ROI"].unique():
        derivative = derivative_df.loc[derivative_df["ROI"] == roi, "Derivative"]
        interpolator = scipy.interpolate.interp1d(frame_times_2p, derivative, kind="linear", bounds_error=False, fill_value="extrapolate")
        roi_upsampled_df = pd.DataFrame({"Frame": cam_frame_indices})
        roi_upsampled_df["ROI"] = roi
        roi_upsampled_df["Derivative"] = interpolator(frame_times)
        upsampled_df = upsampled_df.append(roi_upsampled_df)
    upsampled_df["Date"] = derivative_df["Date"].unique()[0]#get_date(trial_dir)
    upsampled_df["Genotype"] = derivative_df["Genotype"].unique()[0]#get_genotype(trial_dir)
    upsampled_df["Fly"] = derivative_df["Fly"].unique()[0]#get_fly_number(trial_dir)
    upsampled_df["Trial"] = derivative_df["Trial"].unique()[0]#get_trial_number(trial_dir)
    return upsampled_df

def load_fictrac(fictrac_file, date, genotype, fly, trial, filt=None, trajectory=False):
    fictrac_data = utils_ballrot.load_fictrac(fictrac_file, skip_integration=(not trajectory))
    frames = np.arange(len(fictrac_data["delta_rot_forward"]))
    n_frames = len(frames)
    fictrac_index = pd.MultiIndex.from_arrays(
            [
                np.array([date,]  * n_frames),
                np.array([genotype,] * n_frames),
                np.array([fly,] * n_frames),
                np.array([trial,] * n_frames),
                frames,
            ],
            names=("Date", "Genotype", "Fly", "Trial", "Frame")
        )
    fictrac = pd.DataFrame(index=fictrac_index)
    if filt is None:
        filt = scipy.signal.butter(10, 1, btype="lowpass", fs=100, output="sos")
    fictrac["vel"] = fictrac_data["delta_rot_forward"]
    fictrac["side"] = fictrac_data["delta_rot_side"]
    fictrac["turn"] = fictrac_data["delta_rot_turn"]
    fictrac["heading"] = fictrac_data["heading"]
    fictrac["filt_vel"] = scipy.signal.sosfiltfilt(filt, fictrac_data["delta_rot_forward"])
    fictrac["filt_turn"] = scipy.signal.sosfiltfilt(filt, fictrac_data["delta_rot_turn"])
    fictrac["filt_side"] = scipy.signal.sosfiltfilt(filt, fictrac_data["delta_rot_side"])
    fictrac["conv_vel"] = convolve_with_crf(frames / 100, fictrac_data["delta_rot_forward"], None)
    fictrac["conv_turn"] = convolve_with_crf(frames / 100, fictrac_data["delta_rot_turn"], None)
    fictrac["conv_side"] = convolve_with_crf(frames / 100, fictrac_data["delta_rot_side"], None)
    if trajectory:
        fictrac["trajectory_x"] = fictrac_data["x"]
        fictrac["trajectory_y"] = fictrac_data["y"]
    fictrac = fictrac.reset_index()
    return fictrac

def get_residuals(df, regressors, bounds):
    for (roi, trial), sub_df in df.groupby(["ROI", "Trial"]):
        endog = sub_df["dFF"].values
        exog = sub_df[regressors].values
        prediction = predict(np.copy(exog), np.copy(endog), np.copy(exog), bounds=np.copy(bounds))
        df.loc[(df["ROI"] == roi) & (df["Trial"] == trial), "residual_dFF"] = endog - prediction
    return df

def orthogonalize(x, y):
    return x - np.dot(x, y) / np.dot(y, y) * y

def load_antenna_touches(touches_file, date, genotype, fly, trial):
    with open(touches_file, "rb") as fh:
        touches = pickle.load(fh)
    n_frames = len(touches["LArista"])
    frames = np.arange(n_frames)
    df_index = pd.MultiIndex.from_arrays(
            [
                np.array([date,]  * n_frames),
                np.array([genotype,] * n_frames),
                np.array([fly,] * n_frames),
                np.array([trial,] * n_frames),
                frames,
            ],
            names=("Date", "Genotype", "Fly", "Trial", "Frame")
        )
    df = pd.DataFrame(index=df_index)
    df["L antenna touch"] = touches["LArista"]
    df["R antenna touch"] = touches["RArista"]
    df["Assym antenna touch"] = touches["assym"]
    return df

def balance_df(df, col, trials):
    df_ = pd.DataFrame()
    for trial, trial_df in df.groupby("Trial"):
        n = trial_df[col].value_counts().min()
        df_ = df_.append(trial_df.groupby(col).apply(lambda x: x.sample(n)))
    df_.index = df_.index.droplevel(0)
    return df_

def get_beh_rois(results_file, date, fly):
    matrix_results = pd.read_csv(results_file)
    matrix_results = matrix_results.loc[matrix_results["Variable"].isin(BEHAVIOUR_CLASSES), :]
    matrix_results_fly = matrix_results.loc[matrix_results["Fly"] == f"{date}_{fly}", :]
    matrix_results_fly = matrix_results_fly.sort_values("rsquared", ascending=False)
    matrix_results_fly = matrix_results_fly.drop_duplicates("ROI", keep="first")
    matrix_results_fly = matrix_results_fly[["Variable", "ROI", "rsquared"]]
    matrix_results_fly = matrix_results_fly.rename(columns={"Variable": "Behaviour"})
    return matrix_results_fly

def plot_kernel_density_estimate(output_path, groups, colors, x=np.arange(640), bandwidth=50, midline=None, all_data=None):
    kd_grid = sklearn.model_selection.GridSearchCV(
                sklearn.neighbors.KernelDensity(kernel = 'gaussian'),
                {'bandwidth': bandwidth}, cv = 5, n_jobs=-1,
              )
    if all_data is None:
        normalization = np.ones_like(x)
    else:
        kde = kd_grid.fit(all_data[:, np.newaxis])
        log_dens = kde.score_samples(x[:, np.newaxis])
        normalization = np.exp(log_dens)
        normalization /= np.sum(normalization)
    
    plt.figure(figsize=(1, 0.4))
    for points, color in zip(groups, colors):
        kde = kd_grid.fit(points[:, np.newaxis])
        log_dens = kde.score_samples(x[:, np.newaxis])
        y = np.exp(log_dens)
        y = y / normalization
        y /= np.sum(y)
        plt.plot(x, y, color=color)
        fill_color = list(color)
        fill_color[3] = 0.2
        plt.fill_between(x, y, color=fill_color)
    plt.xlim((np.min(x), np.max(x)))
    if midline is not None:
        plt.axvline(x=midline, color="black", linestyle="--")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axes.xaxis.set_ticks([])
    plt.ylim(0, 0.0045)
    plt.savefig(output_path, transparent=True)
    plt.close()

def reorder_ROIs(df, order_csv):
    order_beh = pd.read_csv(order_csv, index_col=0).reset_index()
    if "Fly_ROI" not in df.columns:
        df["Fly_ROI"] = df["Date"].astype(str) + "_" + df["Fly"].astype(str) + " " + df["ROI"].astype(str)
    df = df.merge(order_beh[["Fly_ROI", "New_ROI"]], on="Fly_ROI")
    df["Old_ROI"] = df["ROI"]
    df["ROI"] = df["New_ROI"].astype(int)
    df = df.drop(columns="New_ROI")
    return df

def plot_kde_2D(x, y, output_file):
    values = np.vstack([y, x])
    kernel = scipy.stats.gaussian_kde(values)

    X, Y = np.mgrid[0:256:256j, 0:640:640j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    plt.figure(figsize=(2.5, 1))
    plt.axis("off")
    plt.imshow(Z, vmin=0, vmax=2.5e-5)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(output_file)
    plt.close()

def hysteresis_filter(seq, n=5, n_false=None):
    """
    This function implements a hysteresis filter for boolean sequences.
    The state in the sequence only changes if n consecutive element are in a different state.

    Parameters
    ----------
    seq : 1D np.array of type boolean
        Sequence to be filtered.
    n : int, default=5
        Length of hysteresis memory.
    n_false : int, optional, default=None
        Length of hystresis memory applied for the false state.
        This means the state is going to change to false when it encounters
        n_false consecutive entries with value false.
        If None, the same value is used for true and false.

    Returns
    -------
    seq : 1D np.array of type boolean
        Filtered sequence.
    """
    if n_false is None:
        n_false = n
    #seq = seq.astype(np.bool)
    state = seq[0]
    start_of_state = 0
    memory = 0

    current_n = n
    if state:
        current_n = n_false
    
    for i in range(len(seq)):
        if state != seq[i]:
            memory += 1
        elif memory < current_n:
            memory = 0
            continue
        if memory == current_n:
            seq[start_of_state:i - current_n + 1] = state
            start_of_state = i - current_n + 1
            state = not state
            if state:
                current_n = n_false
            else:
                current_n = n
            memory = 0
    seq[start_of_state:] = state
    return seq
