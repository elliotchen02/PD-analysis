from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import combinations
import joblib
import numpy as np
import pandas as pd
import os

from utils.utils import find_period
from handExtractionUpdated import extract_hands


FPS = 59
COVERAGE = 0.5


def preprocess_landmarks(extraction_dict):
    """
    Input:
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
    Input:
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
    See get_thumb_index_dis
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



def extract_hand_turning(hand_landmarks, plot=False, title=""):
    """
    Analyzes hand turning by measuring the distance of thumb finger movement. 
    Plots data. 
    --------------------------------------------------------------------
    Input:
        hand_landmarks : a processed array returned by preprocess_landmarks()
        plot : (bool) indicating whether to plot
        title : (string) selected title for graph
    Output:
        output : (List)

    """
    output = []
    # using top third of thumb finger as the basis
    hand_land_sum = hand_landmarks[:, :, [4, 3], 0][0].sum(axis=1)
    # tracking the rotation through the top third of index finger
    period, _ = find_period(hand_land_sum)
    peaks_indx, _ = find_peaks(hand_land_sum, height=np.mean(hand_land_sum),
                        distance=int(period * 0.4))

    if plot:
        fig, ax = plt.subplots()
        ax.plot(hand_land_sum)
        plt.grid(which='major', axis='both')
        ax.set_xlabel("Identified Frame")
        ax.plot(peaks_indx, np.array(hand_land_sum)[peaks_indx], "x", ms=10)
        ax.set_ylabel("Finger x-axis displacement")
        ax.set_title(title)
        plt.show()

    h = np.diff(peaks_indx, prepend=0).mean() / FPS
    h_d = (np.diff(peaks_indx, prepend=0)[:4].mean() / FPS - np.diff(peaks_indx, prepend=0)[-5:].mean() / FPS) / 2
    
    output.append((h, h_d))

    return output


def extract_hand_turning_all(hand_landmarks, plot=False, title=""):
    """
    Extract hand turning for all 21 landmark indices in hand_landmarks.
    """
    
    output = []
    
    for finger_idx in range(21):
        hand_land_sum = hand_landmarks[:, :, finger_idx, 0][0].sum(axis=1)
        
        period, _ = find_period(hand_land_sum)
        peaks_indx, _ = find_peaks(hand_land_sum, height=np.mean(hand_land_sum),
                            distance=int(period * 0.4))

        if plot:
            _, ax = plt.subplots()
            ax.plot(hand_land_sum)
            plt.grid(which='major', axis='both')
            ax.set_xlabel("Identified Frame")
            ax.plot(peaks_indx, np.array(hand_land_sum)[peaks_indx], "x", ms=10)
            ax.set_ylabel("Finger x-axis displacement")
            ax.set_title(title)
            plt.show()

        h = np.diff(peaks_indx, prepend=0).mean() / FPS
        h_d = (np.diff(peaks_indx, prepend=0)[:4].mean() / FPS - np.diff(peaks_indx, prepend=0)[-5:].mean() / FPS) / 2
        
        output.append((h, h_d))

    return output

#TODO
def extract_hand_turning_v_lm(hand_pose, ax=None, title="", lms=[4, 8, 16, 20], pick=4):
     output = []
     comb = combinations(lms, pick)

     for i, c in enumerate(comb):

         h_p = hand_pose[:, list(c), 0].sum(axis=1)
         T, _ = find_period(h_p)
         p, values = find_peaks(h_p, height=np.mean(h_p),
                                distance=int(T * 0.4))

         if ax:
             ax.plot(h_p)
             ax.set_xlabel("Frame")
             ax.plot(p, np.array(h_p)[p], "x", ms=10)
             ax.set_ylabel("finger x-axis displacement")
             ax.set_title(title)

         h, h_d = np.diff(p, prepend=0).mean() / fps, (
                     np.diff(p, prepend=0)[:4].mean() / fps - np.diff(p, prepend=0)[-5:].mean() / fps) / 2
         output.append(h)
         output.append(h_d)

     return output, i

#TODO
def hand_feature_extraction(path_ls):
    date_ls = []
    pid_ls = []
    out_result = []
    all_hand_id = ["AR", "AL", "BR", "BL"]
    hand_id = ["Right", "Left"]

    for e, path in enumerate(path_ls):
        date = path.split("_")[0]
        pid = path.split("_")[1]
        all_path = [f"../handOutput3/{date}_{pid}_{hid}_hand.txt" for hid in all_hand_id]
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        out_ls = []

        if pid[-1] in ["A", "a", "B", "b"]:
            dt = joblib.load(f"../handOutput3/{date}_{pid}_hand.txt")
            hand_pose_arr = preprocess_landmarks(dt)[:, 0, :, :]
            q = hand_pose_arr.shape[0] // 4
            t1 = q
            t2 = 2 * q
            t3 = 3 * q

            try:
                hf1 = list(extract_thumb_index_periods(hand_pose_arr[5 * 59:10 * 59, :, :], ax=ax[0, 0],
                                                       title=f"Thumb-index {hand_id[0]}"))
            except:
                hf1 = (np.ones((3)) * np.nan).tolist()
            try:
                hf2 = list(extract_thumb_index_periods(hand_pose_arr[15 * 59:20 * 59, :, :], ax=ax[0, 1],
                                                       title=f"Thumb-index {hand_id[1]}"))
            except:
                hf2 = (np.ones((3)) * np.nan).tolist()
            try:
                hf3 = list(extract_hand_turning(hand_pose_arr[t2 * 59:t2 + 5 * 59, :, :], ax=ax[1, 0],
                                                title=f"Thumb-Hand_turning {hand_id[0]}"))
            except:
                hf3 = (np.ones((2)) * np.nan).tolist()
            try:
                hf4 = list(extract_hand_turning(hand_pose_arr[t3 * 59:t3 + 5 * 59, :, :], ax=ax[1, 1],
                                                title=f"Thumb-Hand_turning {hand_id[1]}"))
            except:
                hf4 = (np.ones((2)) * np.nan).tolist()
            ax[0, 0].text(0, 0.5, str(4 * q))
            out_ls = hf1 + hf2 + hf3 + hf4

        else:
            for i, all_hand_path in enumerate(all_path):
                if os.path.isfile(all_hand_path):

                    dt = joblib.load(all_hand_path)
                    hand_pose_arr = preprocess_landmarks(dt)[:, 0, :, :]

                    if i <= 1:
                        hf = extract_thumb_index_periods(hand_pose_arr, ax=ax[0, i], title=f"Thumb-index {hand_id[i]}")
                        out_ls += list(hf)
                    else:
                        hf = extract_hand_turning(hand_pose_arr, ax=ax[1, i % 2], title=f"Hand_turning {hand_id[i % 2]}")
                        out_ls += list(hf)

                else:
                    if i <= 1:
                        out_ls += (np.ones((3)) * np.nan).tolist()
                    else:
                        out_ls += (np.ones((2)) * np.nan).tolist()

        out_result.append(out_ls)
        date_ls.append(date)
        pid_ls.append(pid)

        plt.savefig(f"../handOutput3/outfig/{date}_{pid}.png")
        plt.close()

        df = pd.DataFrame(
            np.concatenate([np.array(date_ls).reshape(-1, 1), np.array(pid_ls).reshape(-1, 1), np.array(out_result)],
                           axis=1))

        col = df.columns.tolist()
        col[0] = "Date"
        col[1] = "PID"

        df.columns = col
        df.Date = df.Date.apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[-2:]}")
        df.PID = df.PID.astype(int)

        return df

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
    
    print(extract_hand_turning(hand_pose_array, plot=True))


    


    