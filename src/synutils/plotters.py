from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def legend_without_duplicate_labels(ax: plt.Axes):
    """remove duplicate labels from legend

    Args:
        ax ([plt.Axes]): axes object
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot_projections(
    axes: np.ndarray,
    cls_label: Optional[np.ndarray] = None,
    marker_size: int = 3,
    cmap_name: str = "Set3",
) -> plt.Figure:
    """plot projected axes of the datapoints
    each column is plotted to a subplot, versus the first column.

    Args:
        axes (np.ndarray): vector or arraylike, each column is plotted to a subplot.
        cls_label (np.ndarray, optional): same shape as val array. class labels per row, defaults to None.
        cmap_name (str, optional): name of the colormap, defaults to "hsv".
            more colormaps can be found at https://matplotlib.org/stable/tutorials/colors/colormaps.html
        marker_size (int, optional): size of the marker, defaults to 3.

    Returns:
        plt.Figure: plot figure
    """
    unique_labels = np.unique(cls_label)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    cmap = cmap(np.linspace(0, 1, len(unique_labels)))
    fig = plt.figure(figsize=(20 * (axes.shape[1] - 1), 20))
    for ii in range(1, axes.shape[1]):
        ax = fig.add_subplot(1, axes.shape[1] - 1, ii)
        for jj, ulabel in enumerate(unique_labels):
            idx = cls_label == ulabel
            ax.scatter(
                axes[idx, 0],
                axes[idx, ii],
                s=marker_size,
                color=cmap[jj],
                alpha=0.3,
                label=ulabel,
            )
        ax.set_xlabel("axis 1")
        ax.set_ylabel(f"axis {ii+1}")
        ax.set_title(f"axis 1 vs axis {ii+1}")
    plt.tight_layout()
    plt.legend()
    return fig
