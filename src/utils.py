import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Callable
import time


def plot_confusion(confusion_matrix: np.ndarray, title=None):
    fontsize = 40

    sns.set(font_scale=1.5)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        norm=LogNorm(),
    )
    if title:
        plt.title(title)
    else:
        plt.title("Confusion matrix")

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()


def timeit(func: Callable, *args, **kwargs):
    """Function to time another function

    Parameters:
        func: function to be timed
        args: positional arguments for function
        kwargs: keyword arguments for function

    Returns:
        average number of seconds taken for the function to run
    """
    n = 5
    avg = 0
    for i in range(n):
        start = time.time()
        func(*args)
        avg += (time.time() - start) / n

    print(f"Average time taken for {n} calls: {avg}")
    return avg


def conf_mat(preds, labl, num_cls):
    return tf.math.confusion_matrix(labels=labl, predictions=preds, num_classes=num_cls)


def perc(matrix):
    conf = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            conf[i][j] = matrix[i][j] / tf.reduce_sum(matrix[i], 0).numpy()
    return conf
