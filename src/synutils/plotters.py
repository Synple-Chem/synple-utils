from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_projections(
    axes: np.ndarray, cls_label: Optional[np.ndarray] = None
) -> plt.Figure:
    """plot projected axes of the datapoints
    each column is plotted to a subplot.

    Args:
        val (np.ndarray): vector or arraylike, each column is plotted to a subplot.
        cls_label (np.ndarray, optional): same shape as val array. class labels per row, defaults to None.

    Returns:
        plt.Figure: plot figure
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.scatter(val_pca[:, 0], val_pca[:, 1], c=cls_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2")
    ax = fig.add_subplot(122)
    ax.scatter(val_pca[:, 0], val_pca[:, 2], c=cls_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC3")
    ax.set_title("PC1 vs PC3")
    plt.tight_layout()
    return fig
