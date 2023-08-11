from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import combinations
import joblib
import numpy as np
import os

from utils.featurePlot import plot_hand_features
from utils.utils import find_period
from handExtractionUpdated import extract_hands


FPS = 59
COVERAGE = 0.5


def preprocess_landmarks(extraction_dict):
    """
    Args:
        extraction_dict: Dictionary recieved from extract_hands
    Output:
        processed_array: List of landmark arrays with structure:
            [ 
                [ List of Hand landmarkers by frame ]
                        .
                        .
                        .
            ]
        processed_array is an multi-dim array. Each index is a list corresponding to a 
        specific hand's landmarkers for all frames.
    """
    processed_array = []
    frame_by_frame = extraction_dict['landmarks']
    for frame in frame_by_frame:
        if len(frame) != 0:
            for hand_idx, landmark_array in frame:
                try:
                    processed_array[hand_idx].append(landmark_array)
                except Exception:
                    processed_array.append([])
                    processed_array[-1].append(landmark_array)

    return np.array(processed_array)


def get_thumb_index_dis(hand_landmarks):
    """
    Args:
        hand_landmarks: List containing 2D numPy array of landmark coordinates for every frame
    Output:
        hand_4_8_dis: Distance between thumb and index finger tip for every frame
    """

    # Finds the relative positioning of thumb and index finger (4 and 8 respectively as per MediaPipe model)
    hand_thumb = hand_landmarks[:, :, 4, :-1]   #TODO only taking x,y and not z values? Unclear...
    hand_index = hand_landmarks[:, :, 8, :-1]

    print(hand_thumb)
    hand_4_8_dis = []
    
    # Need to zip each frame's landmarks not entire landmark matrix (hence nested loops)
    for hand_num in range(len(hand_thumb)):
        for t, i in zip(hand_thumb[hand_num], hand_index[hand_num]):
            hand_4_8_dis.append(np.linalg.norm(i - t))
    hand_4_8_dis = np.array(hand_4_8_dis)
    
    return hand_4_8_dis


def get_thumb_pinky_dis(hand_landmarks):
    """
    See get_thumb_index_dis.
    --------------------------------------
    """
    hand_thumb = hand_landmarks[:, :, 4, :-1]
    hand_index = hand_landmarks[:, :, 20, :-1]

    hand_4_20_dis = []
    for hand_num in range(len(hand_thumb)):
        for t, i in zip(hand_thumb[hand_num], hand_index[hand_num]):
            hand_4_20_dis.append(np.linalg.norm(i - t))
    hand_4_20_dis = np.array(hand_4_20_dis)

    return hand_4_20_dis


#TODO
def extract_thumb_index_periods(hand_pose, ax=None, title=""):
    hand_dis = get_thumb_index_dis(hand_pose)
    mean_hand_dis = np.mean(hand_dis)
    hand_dis = hand_dis / hand_dis.max()

    T, _ = find_period(hand_dis)
    p, values = find_peaks(hand_dis, height=np.mean(hand_dis),
                           distance=int(T * 0.4))

    if ax:
        ax.plot(hand_dis)
        ax.plot(p, np.array(hand_dis)[p], "x", ms=10)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Relative thumb-index distance")
        ax.set_title(title)

    return np.diff(p, prepend=0).mean() / FPS, (
                np.diff(p, prepend=0)[:4].mean() / FPS - np.diff(p, prepend=0)[-5:].mean() / FPS) / 2, mean_hand_dis


def extract_hand_turning(hand_landmarks, landmark_index=21, plot=False):
    """
    Analyzes hand turning by measuring the distance of thumb finger movement. 
    Plots data. 
    --------------------------------------------------------------------
    Args:
        hand_landmarks : a processed array returned by preprocess_landmarks()
        landmark_index: selects which finger to analyze. Default is all 21 hand landmarks. 
        plot : indicating whether to plot
        title : selected title for graph
    Output:
        output : (List)

    """
    output = []

    for finger_idx in range(1, landmark_index + 1):
        hand_land_sum = hand_landmarks[:, :, [finger_idx - 1, finger_idx], 0][0].sum(axis=1)
        print(hand_land_sum)

        period, _ = find_period(hand_land_sum)
        peaks_indx, _ = find_peaks(
                            hand_land_sum, 
                            height=np.mean(hand_land_sum),
                            distance=int(period * 0.4)
                        )

        if plot:
            plot_hand_features(
                hand_land_sum=hand_land_sum,
                peaks_indx=peaks_indx,
                title=f'Current Finger Index: {finger_idx}',
                x_label='Identified Frame',
                y_label='Finger x-axis Displacement',
                save=True
            )

        # h = mean of each peaks duration (second)
        # h_d = the differences of last 5 peaks compared to the 1st five peaks
        h = np.diff(peaks_indx, prepend=0).mean() / FPS
        h_d = (np.diff(peaks_indx, prepend=0)[:4].mean() / FPS - np.diff(peaks_indx, prepend=0)[-5:].mean() / FPS) / 2
        
        output.append((h, h_d))

    return output


#TODO
def single_thumb_index_hand(r_path, l_path, out_dir):

    fig, ax = plt.subplots(2, 1, figsize=(20, 20))
    r_dt = joblib.load(r_path)
    right_hand_arr = preprocess_landmarks(r_dt)[:, 0, :, :]
    r_hf = extract_thumb_index_periods(right_hand_arr, ax=ax[0], title=f"Thumb-index right hand")

    l_dt = joblib.load(l_path)
    left_hand_arr = preprocess_landmarks(l_dt)[:, 0, :, :]
    l_hf = extract_thumb_index_periods(left_hand_arr, ax=ax[1], title=f"Thumb-index left hand")

    plt.savefig(f"{out_dir}vis_hand_features_extraction_.png")
    plt.close()

    return list(r_hf) + list(l_hf)


if __name__ == '__main__':
    video_path = '/Users/elliot/Documents/NTU 2023/PDAnalysis/20200702_9BL.mp4'
    hand_pose_array = preprocess_landmarks(extract_hands(video_path))
    print(hand_pose_array)
    print(extract_hand_turning(
            hand_landmarks=hand_pose_array, 
            plot=True))


    


    
