from .gaitExtraction import gait_extraction
from .old_analysis.handExtraction import hand_extraction
from .voiceFeatureExtraction import voice_features_extraction
from .old_analysis import handFeaturesExtraction
from . import gaitFeaturesExtraction
import os
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


MODEL_PATHS = "/home/pdapp/pd_api_server/api/pdModel/PD_pretrained_models/"
TRAINED_MODELS_LS = ["C4.5 DT", "RF", "LogReg", "AdaBoost", "NB", "SVM"]  # remove KNN, "GBM", "LightGBM",

gait_feature_name = ['left_foot_ground', 'right_foot_ground', 'left_right_foot_len_average', 'left_right_foot_len_max',
                     'left_turning_duration', 'left_turning_slope', 'right_turning_duration', 'right_turning_slope',
                     'l_leg_max_angles', 'l_leg_min_angles', 'r_leg_max_angles', 'r_leg_min_angles', 'l_arm_max_angles',
                     'l_arm_min_angles', 'r_arm_max_angles', 'r_arm_min_angles', 'core_max_angles', "core_min_angles",
                     'average_duration_per_rounds', "duration_change", "l_mean_steps", "r_mean_steps"]

hand_features_name = ["right_finger_open_duration", 'right_finger_open_change', "right_finger_open_distance",
                      "left_finger_open_duration", 'left_finger_open_change', "left_finger_open_distance"]

voice_feature_name = ['pause(%)', 'volume change', 'pitch change', 'Average pitch']


def extract_gait(gait_video_pth, out_dir):
    gait_extraction(gait_video_pth, out_dir)
    pth_2d = f"{out_dir}2d_{os.path.basename(gait_video_pth)[:-4]}.npy"
    pth_3d = f"{out_dir}3d_{os.path.basename(gait_video_pth)[:-4]}.npz"
    gait_feature = gaitFeaturesExtraction.pose_features_extract(pth_2d, pth_3d, plot_results=True,
                                                                save_fig_pth=f"{out_dir}vis_gait_extraction.png")
    return gait_feature


def extract_hand(hand_video_pth, out_dir, hand):
    hand_extraction(hand_video_pth, out_video_root=out_dir, hand=hand)


# to complete, out_dir? 
def extract_voice(voice_file_pth):
    return voice_features_extraction(voice_file_pth)


def model_extraction(gait_video_pth, left_video_path, right_video_path, voice_path, out_dir):
    print(f'Start gait features extraction')
    gait_feature = extract_gait(gait_video_pth, out_dir)
    print(f'Start left hand features extraction')
    extract_hand(left_video_path, out_dir, 'left')
    print(f'Start right hand features extraction')
    extract_hand(right_video_path, out_dir, 'right')

    r_path = f"{out_dir}right_hand_{os.path.basename(right_video_path)[:-4]}.txt"
    l_path = f"{out_dir}left_hand_{os.path.basename(left_video_path)[:-4]}.txt"

    hand_feature = handFeaturesExtraction.single_thumb_index_hand(r_path, l_path, out_dir)
    print(f"Finish hand features extraction")

    voice_feature = voice_features_extraction(voice_path)

    print(f"{gait_feature}, \n {hand_feature}, \n {voice_feature} \n")

    all_features = gait_feature + hand_feature + voice_feature
    np.save(f'{out_dir}all_feature.npy', all_features)
    print(f"save! \n {all_features}")


def predict_models(all_features_pth, age, gender, out_dir):
    sfs_idx = joblib.load(f'{MODEL_PATHS}nvp_sfs_idx.txt')
    _temp_voice_sfs_idx = joblib.load(f'/home/pdapp/voice_noscore_sfs.txt')
    gait_len = len(gait_feature_name)
    hand_len = len(hand_features_name)
    voice_len = len(voice_feature_name)
    gait_result = {}
    hand_result = {}
    voice_result = {}
    all_result = {}

    for r in [gait_result, hand_result, voice_result, all_result]:
        for k in TRAINED_MODELS_LS:
            r[k] = [0, 0]

    data = np.load(all_features_pth)
    gait_feature = np.concatenate([np.array([age, gender]), data[:gait_len]])
    hand_feature = np.concatenate([np.array([age, gender]), data[gait_len:gait_len+hand_len]])
    voice_feature = np.concatenate([np.array([age, gender]), data[-voice_len:]])

    gait_result = deploy(np.array([gait_feature]), sfs_idx['gait_sfs_idx'], modal="gait", fold=10)
    hand_result = deploy(np.array([hand_feature]), sfs_idx['hand_sfs_idx'], modal="hand", fold=10)
    voice_result = deploy(np.array([voice_feature]), _temp_voice_sfs_idx, modal="voice2", fold=10)
    weight = [4, 6, 5]

    all_result = {}

    for k in TRAINED_MODELS_LS:
        all_result[k] = np.average(np.array([[gait_result[k]], [hand_result[k]],
                                          [voice_result[k]]]),  weights=weight, axis=0)[0]


    # plot results
    results = np.array([gait_result["SVM"][0], hand_result["SVM"][0], voice_result["SVM"][0],
                                               all_result["SVM"][0]])*100
    plt.bar(["Gait", "Hand", "Voice", "All"], results)

    for i, r in enumerate(results):
        plt.text(i - 0.2, 105, f"{r:.2f}%")

    for i, s in enumerate(["Gait", "Hand", "Voice", "All"]):
        plt.text(i - 0.2, 115, f"{s}")

    plt.ylim([0, 120])
    plt.axis("off")
    plt.savefig(f"{out_dir}result.png")

    return results


def deploy(data, feature_idx_dt, modal="gait", fold=10):
    save_file_dir = f"{MODEL_PATHS}Save_model_{modal}/"
    proba_out_dt = {}

    for model_name in TRAINED_MODELS_LS:
        proba_ls = []

        for i in range(fold):
            clf_pth = f'{save_file_dir}{i}_{model_name}.joblib'
            clf = joblib.load(clf_pth)
            print(clf.n_features_in_)
            f_idx = feature_idx_dt[model_name]
            predict_proba = clf.predict_proba(data[:, f_idx])
            proba_ls.append(predict_proba)

        mean_test = np.array(proba_ls).mean(0)
        proba_out_dt[model_name] = mean_test[:, 1]

    return proba_out_dt


if __name__ == "__main__":
    gait_v_pth = "/mnt/pd_app/walk/20200818_5C.mp4"
    left_v_path = "/mnt/pd_app/gesture/202008069_AL.mp4"
    right_v_path = "/mnt/pd_app/gesture/202008069_AR.mp4"
    out_d = "/mnt/pd_app/results/test/"

    ini_time_for_now = datetime.now()
    asyncio.run(model_extraction(gait_v_pth, left_v_path, right_v_path, out_d))
    finish_time = datetime.now()
    print((finish_time - ini_time_for_now).seconds)

    # data = np.load(f'{out_d}all_feature.npy')
    # deploy(data, feature_idx_dt, modal="gait", fold=10)

