"""
This module provides a function for plotting a list of images
along with their optional labels.


Functions:
- plot_images(images, plot_labels=None, figsize=None, max_cols=1):
        Plots a list of images along with their optional labels.

"""

import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_images(images: list | np.ndarray,
                plot_labels: list | np.ndarray = None,
                figsize: tuple = None, max_cols: int = 1):
    """
    Plots a list of images along with their optional labels.

    Parameters:
    ------------
    - images (list or np.ndarray):
            A list of images or a single image as a numpy array.
    - plot_labels (list or np.ndarray, optional):
            A list of labels corresponding to the images. Default is None.
    - figsize (tuple, optional):
            The size of the figure. Default is None.
    - max_cols (int, optional):
            The maximum number of columns in the plot. Default is 1.

    Returns:
    ------------
    None

    Raises:
    ------------
    - Warning: If the number of images and labels are not the same.

    Example usage:
    ------------

    images = [image1, image2, image3]
    labels = ['Label 1', 'Label 2', 'Label 3']
    plot_images(images, plot_labels=labels, figsize=(10, 5), max_cols=2)
    """
    if not isinstance(images, list):
        images = [images]

    if plot_labels is not None:
        if not isinstance(plot_labels, list):
            plot_labels = [plot_labels]

        if len(images) != len(plot_labels):
            plot_labels = None
            warnings.warn("Number of images and labels are not the same. "
                          "Plotting images without labels.")

    fig = plt.figure()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)

    num_images = len(images)
    num_cols = min(max_cols, num_images)
    num_rows = num_images // num_cols

    if num_images % num_cols != 0:
        num_rows += 1

    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(images[i])
        ax.axis("off")

        if plot_labels is not None:
            ax.set_title(plot_labels[i])

    plt.tight_layout()
    plt.show()
