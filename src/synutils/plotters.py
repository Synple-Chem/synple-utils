from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_projections(
    axes: np.ndarray, cls_label: Optional[np.ndarray] = None
) -> plt.Figure:
    """plot projected axes of the datapoints
    each column is plotted to a subplot, versus the first column.

    Args:
        axes (np.ndarray): vector or arraylike, each column is plotted to a subplot.
        cls_label (np.ndarray, optional): same shape as val array. class labels per row, defaults to None.

    Returns:
        plt.Figure: plot figure
    """
    fig = plt.figure(figsize=(3 * (axes.shape[1] - 1), 3))
    for ii in range(1, axes.shape[1]):
        ax = fig.add_subplot(1, axes.shape[1] - 1, ii)
        ax.scatter(axes[:, 0], axes[:, ii], c=cls_label)
        ax.set_xlabel("axis 1")
        ax.set_ylabel(f"axis {ii+1}")
        ax.set_title(f"axis 1 vs axis {ii+1}")
    plt.tight_layout()
    return fig
