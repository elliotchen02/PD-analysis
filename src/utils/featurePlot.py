import numpy as np
import matplotlib.pyplot as plt

def plot_hand_features(hand_land_sum, peaks_indx, title='', x_label = '', y_label=''):
    """
    Plots features for hand extraction. 
    ----------------------------------------
    Args:
        hand_land_sum: processed hand landmarks
        peaks_indx: peaks of periods for hands
        title: title of plot
        x_label: x-axis title
        y_label: y-axis title
    """
    _, ax = plt.subplots()
    ax.plot(hand_land_sum)
    plt.grid(which='major', axis='both')
    ax.set_xlabel(x_label)
    ax.plot(peaks_indx, np.array(hand_land_sum)[peaks_indx], "x", ms=10)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()