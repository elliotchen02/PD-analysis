import numpy as np
import matplotlib.pyplot as plt

def plot_hand_features(hand_land_sum: np.array, 
                       peaks_indx: int, 
                       title: str='', 
                       x_label: str = '', 
                       y_label: str='', 
                       save: bool=False) -> None:
    fig, ax = plt.subplots()
    ax.plot(hand_land_sum)
    plt.grid(which='major', axis='both')
    ax.set_xlabel(x_label)
    ax.plot(peaks_indx, np.array(hand_land_sum)[peaks_indx], "x", ms=10)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

    if save == True:
        fig.savefig(f'/Users/elliot/Documents/NTU 2023/PDAnalysis/extractions/plots/raw/{title}.png')