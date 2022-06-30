import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import PathCollection
from typing import Tuple, List, Optional, Sized, Dict, Union

def hist_weights(
    weight_collections,
    ax,
    figsize: Tuple[float, float] = (8.0, 4.5),
    bins = 20
):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax.clear()

    n, bins, patches = ax.hist(weight_collections, bins)

    ax.set_xlabel("Range of weights")
    ax.set_ylabel("Counts of synapses")
    ax.set_title("Weights of synapses")

    return  ax

