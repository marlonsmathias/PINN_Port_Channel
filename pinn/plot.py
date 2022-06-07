from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, Axes


def figsize(scale: float, nplots = 1) -> Tuple[float, float]:
    """
    Create figsize based on scale and nplots

    Args:
        scale (float): Scale.
        nplots (int, optional): number of plots. Defaults to 1.

    Returns:
        Tuple[float, float]: figsize tuple
    """
    fig_width_pt = 390.0
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = nplots*fig_width*golden_mean
    fig_size = [fig_width,fig_height]
    return fig_size


pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        ]
}


def newfig(width, nplots = 1) -> Tuple[Figure, Axes]:
    """
    Create new figure.

    Args:
        width ([type]): width of image
        nplots (int, optional): number of plots per image. Defaults to 1.

    Returns:
        Tuple[Figure, Axes]: new figure with its respective axis.
    """
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(fig: Figure, filename: str):
    """
    save figure.

    Args:
        fig (Figure): figure to save.
        filename (str): path/filename to save figure.
    """
    fig.savefig(filename, dpi=fig.dpi)
