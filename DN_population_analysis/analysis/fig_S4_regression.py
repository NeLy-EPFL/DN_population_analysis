import numpy as np
import pandas as pd
import tqdm

import DN_population_analysis.utils as utils


directories = utils.load_exp_dirs("../../recordings.txt")
results = pd.DataFrame()

print("Loading data")
for fly_dir, trial_dirs in utils.group_by_fly(directories).items():
    date = utils.get_date(fly_dir)
    fly = utils.get_fly_number(fly_dir)

    fly_df = utils.load_fly_data(trial_dirs, dFF=True, behaviour=True, fictrac=True, angles=True, odor=True)
    fly_df = fly_df.loc[fly_df["Odor"].isin(["H2O", "ACV", "MSC"]), :]

    thresh = 0
    fly_df["turn_l"] = fly_df["turn"].clip(lower=thresh) - thresh
    fly_df["turn_r"] = (fly_df["turn"].clip(upper=thresh) - thresh) * -1

    for roi in tqdm.tqdm(fly_df["ROI"].unique()):
        roi_df = fly_df.loc[fly_df["ROI"] == roi, :]
        endog = roi_df["dFF"].values

        behaviours = list(fly_df["Behaviour"].unique())

        baseline_exog = []
        behaviour_exog = []
        for trial in roi_df["Trial"].unique():
            trial_mask = (roi_df["Trial"] == trial)
            trial_baseline = trial_mask.values.astype(float)
            baseline_exog.append(trial_baseline)
        for beh in behaviours + ["turn_l", "turn_r", "vel"]:
            if beh == "background":
                continue
            if beh in behaviours:
                regressor = (roi_df["Behaviour"] == beh).values.astype(float)
            else:
                regressor = roi_df[beh].values.astype(float)
            regressor =  utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"])
            behaviour_exog.append(regressor)
        baseline_exog = np.array(baseline_exog).transpose()
        behaviour_exog = np.array(behaviour_exog).transpose()

        for joint in ["all_angles", "front_angles", "middle_angles", "hind_angles", "front_l_angles", "middle_l_angles", "hind_l_angles", "front_r_angles", "middle_r_angles", "hind_r_angles"]:
        #for joint in ["front_angles", "front_positions", "middle_angles", "middle_positions", "hind_angles", "hind_positions"]:
        #for joint in ["front_l_angles", "front_l_positions", "middle_l_angles", "middle_l_positions", "hind_l_angles", "hind_l_positions",
        #              "front_r_angles", "front_r_positions", "middle_r_angles", "middle_r_positions", "hind_r_angles", "hind_r_positions"]:
            print(joint)
            if joint == "all_angles":
                columns_shuffeling = roi_df.filter(like="Angle").columns
                columns_remaining = []
            elif joint == "all_positions":
                columns_shuffeling = roi_df.filter(like="Pose").columns
                columns_remaining = []
            else:
                if "angles" in joint:
                    like_arg = "Angle"
                elif "positions" in joint:
                    like_arg = "Pose"
                else:
                    raise NotImplemented

                if "front" in joint:
                    filt_pair = "F"
                elif "middle" in joint:
                    filt_pair = "M"
                elif "hind" in joint:
                    filt_pair = "H"
                else:
                    raise NotImplemented

                if "_l_" in joint:
                    filt_side = "L"
                elif "_r_" in joint:
                    filt_side = "R"
                else:
                    filt_side = ""
                filt = filt_side + filt_pair
                print(like_arg, filt)
                columns_shuffeling = [c for c in roi_df.filter(like=like_arg).columns if filt in c]
                columns_remaining = [c for c in roi_df.filter(like=like_arg).columns if not filt in c]

            #elif joint == "front_angles":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Angle").columns if "F" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Angle").columns if not "F" in c]
            #elif joint == "middle_angles":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Angle").columns if "M" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Angle").columns if not "M" in c]
            #elif joint == "hind_angles":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Angle").columns if "H" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Angle").columns if not "H" in c]
            #elif joint == "front_positions":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Pose").columns if "F" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Pose").columns if not "F" in c]
            #elif joint == "middle_positions":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Pose").columns if "M" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Pose").columns if not "M" in c]
            #elif joint == "hind_positions":
            #    columns_shuffeling = [c for c in roi_df.filter(like="Pose").columns if "H" in c]
            #    columns_remaining = [c for c in roi_df.filter(like="Pose").columns if not "H" in c]
            #else:
            #    raise NotImplemented
            print()
            print("Shuffeling")
            print(columns_shuffeling)
            print()
            print("Remaining")
            print(columns_remaining)
            joint_shuffeling_exog = []
            joint_intact_exog = []
            for col in columns_shuffeling:
                regressor = roi_df[col].values.astype(float)
                joint_shuffeling_exog.append(utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"]))
            for col in columns_remaining:
                regressor = roi_df[col].values.astype(float)
                joint_intact_exog.append(utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"]))
            #else:
            #    regressor = roi_df["Odor"].isin(["ACV", "MSC"]).values.astype(float)
            #joint_exog = [utils.convolve_with_crf(np.arange(len(regressor)) / 100, regressor, roi_df["Trial"]), ]

            joint_shuffeling_exog = np.array(joint_shuffeling_exog).transpose()
            joint_intact_exog = np.array(joint_intact_exog).transpose()

            shuffled_joint_exog = np.copy(joint_shuffeling_exog)
            np.random.shuffle(shuffled_joint_exog)

            if len(joint_intact_exog) != 0:
                full_exog = np.concatenate((baseline_exog, behaviour_exog, joint_intact_exog, joint_shuffeling_exog), axis=1)
                reduced_exog = np.concatenate((baseline_exog, behaviour_exog, joint_intact_exog, shuffled_joint_exog), axis=1)
                full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * behaviour_exog.shape[1] + [[0, np.inf],] * (joint_intact_exog.shape[1] + joint_shuffeling_exog.shape[1])).transpose()
                reduced_bounds = np.copy(full_bounds)
            else:
                full_exog = np.concatenate((baseline_exog, behaviour_exog, joint_shuffeling_exog), axis=1)
                reduced_exog = np.concatenate((baseline_exog, behaviour_exog, shuffled_joint_exog), axis=1)

                full_bounds = np.array([[np.NINF, np.inf],] * baseline_exog.shape[1] + [[0, np.inf],] * behaviour_exog.shape[1] + [[0, np.inf],] * joint_shuffeling_exog.shape[1]).transpose()
                reduced_bounds = np.copy(full_bounds)
            
            trials = roi_df["Trial"].values

            prediction_plots_dir = f"output/FigS4/prediction_plots_dir_{joint}"
            prediction_plots_dir = None
            results = results.append(utils.cv_rsquared(date, fly, joint, roi, full_exog, endog, reduced_exog, trials, full_bounds, reduced_bounds, one_model_per_trial=True,
                                                       prediction_plots_dir=prediction_plots_dir))

    results.to_csv("output/FigS4/predict_activity_from_joints.csv")

results.to_csv("output/FigS4/predict_activity_from_joints.csv")
