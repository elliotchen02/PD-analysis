from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import combinations
import joblib
import numpy as np
import os

from collections import namedtuple

from utils.featurePlot import plot_hand_features
from utils.utils import find_period
from handExtractionUpdated import Hand, extract_hands, preprocess_landmarks


FPS = 59
COVERAGE = 0.5

TurningFeatures = namedtuple(
    'TurningFeatures',
    ['mean_peak_duration', 'peak_difference']
)

def get_thumb_index_dis(hand_obj: Hand) -> list[float]:
    hand_landmarks = np.array(hand_obj.get_landmarks())

    # Finds the relative positioning of thumb and index finger (4 and 8 respectively as per MediaPipe model)
    hand_thumb = hand_landmarks[:, 4, :-1]
    hand_index = hand_landmarks[:, 8, :-1]

    hand_4_8_dis = []
    
    # Need to zip each frame's landmarks not entire landmark matrix (hence nested loops)
    for t, i in zip(hand_thumb, hand_index):
        hand_4_8_dis.append(np.linalg.norm(i - t))
    hand_4_8_dis = np.array(hand_4_8_dis)
    
    return hand_4_8_dis


def get_thumb_pinky_dis(hand_obj: Hand) -> list[float]:
    hand_landmarks = np.array(hand_obj.get_landmarks())

    # Finds the relative positioning of thumb and index finger (4 and 8 respectively as per MediaPipe model)
    hand_thumb = hand_landmarks[:, 4, :-1]
    hand_pinky = hand_landmarks[:, 20, :-1]

    hand_4_20_dis = []
    
    # Need to zip each frame's landmarks not entire landmark matrix (hence nested loops)
    for t, p in zip(hand_thumb, hand_pinky):
        hand_4_20_dis.append(np.linalg.norm(p - t))
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


def extract_hand_turning(hand_obj: Hand, landmark_index: int=21, plot: bool=False) -> list[TurningFeatures]:
    """
    Analyzes hand turning by measuring the distance of x-axis landmark movement. Plots data. 
   """
    output = []
    hand_landmarks = np.array(hand_obj.get_landmarks())
    print(f'Extracting hand turning features for {hand_obj.get_category()} hand')
    print('---------------------------')
    print(np.shape(hand_landmarks))
    for finger_idx in range(0, landmark_index):
        hand_land_sum = hand_landmarks[:, finger_idx, 0]
       
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
                title=f'Current Finger Index: {finger_idx + 1}',
                x_label='Identified Frame',
                y_label='Finger x-axis Displacement',
                save=True
            )

        # h = mean of each peaks duration (second)
        # h_d = the differences of last 5 peaks compared to the 1st five peaks
        h = np.diff(peaks_indx, prepend=0).mean() / FPS
        h_d = (np.diff(peaks_indx, prepend=0)[:4].mean() / FPS - np.diff(peaks_indx, prepend=0)[-5:].mean() / FPS) / 2
        
        output.append(TurningFeatures(h, h_d))

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
    hands_in_video_l = preprocess_landmarks(extract_hands(video_path))
   
    extract_hand_turning(hands_in_video_l[0], plot=True)

    
    


    
    


    
