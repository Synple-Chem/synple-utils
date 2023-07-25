from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_pc_projections(val: np.ndarray, cls_label: Optional[np.ndarray]=None)-> plt.Figure:
    """plot pc1 and pc2 projection of the datapoints
    each column is plotted to a subplot. 

    Args:
        val (np.ndarray): vector or arraylike, each column is plotted to a subplot. 
        cls_label (np.ndarray, optional): same shape as val array. class labels per row, defaults to None.

    Returns:
        plt.Figure: plot figure
    """

    fig = plt.figure()


    return fig