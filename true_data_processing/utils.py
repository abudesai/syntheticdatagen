import matplotlib.pyplot as plt
import pandas as pd, numpy as np


TITLE_FONT_SIZE = 16

def plot_samples(samples, n, title):    
    fig, axs = plt.subplots(n, 1, figsize=(6,8))
    i = 0
    for _ in range(n):
        rnd_idx = np.random.choice(len(samples))
        s = samples[rnd_idx]
        axs[i].plot(s)    
        i += 1

    fig.suptitle(title, fontsize = TITLE_FONT_SIZE)
    # fig.tight_layout()
    plt.show()