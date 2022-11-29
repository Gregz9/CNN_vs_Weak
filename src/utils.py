import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def PCA_stoch(X, n_components):
    print("eric")

def plot_confusion(confusion_matrix: np.ndarray, title=None):
    fontsize = 40

    sns.set(font_scale=3)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2%",
        cmap="Blues",
    )
    if title:
        plt.title(title)
    else:
        plt.title("Confusion matrix")

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.show()


def PCA_stoch(X, n_components, iterations=10, eta=1e-5, convergence=1e-5, epochs=10):
    print(f"{X.shape=}")
    m, n = X.shape
    W_prev_epoch = np.eye(m, n_components)

    for e in range(epochs):
        print(e)

        W_prev_iter = W_prev_epoch
        u_mark = np.zeros((m, n_components))
        for i in range(n):
            u_mark += np.outer(X[:, i], (X[:, i].T @ W_prev_epoch)) / n

        for t in range(iterations):
            print(t)
            i = np.random.randint(n)

            X_i = X[:, i].reshape(X[:, i].shape[0], 1)
            W_mark = W_prev_iter + eta * (
                X_i * (X_i.T @ W_prev_iter - X_i.T @ W_prev_epoch) + u_mark
            )
            W_prev_iter, _ = np.linalg.qr(W_mark)

        W_prev_epoch = W_prev_iter

    return W_prev_epoch


def PCA_fit(X, n_components):
    if X.shape[0] < n_components:
        raise ValueError("n_components is higher than height of X")

    means = tf.reduce_mean(X, axis=0)
    stds = tf.math.reduce_std(X, axis=0)
    stds = tf.where(tf.equal(stds, 0), tf.ones_like(stds), stds)
    X = (X - means) / stds

    _, _, W = tf.linalg.svd(X)

    return W[:, :n_components]


class PCALayer(layers.Layer):
    def __init__(self, W):
        super(PCALayer, self).__init__()
        self.num_outputs = W.shape[1]
        self.trainable = False
        # self.W = W
        self.W = tf.cast(W, dtype="float32")

    def call(self, inputs):

        return tf.matmul(inputs, self.W)


# PCA implemented using tensors, to be able to run on gpu

# class PCA:
#     def __init__(self, n_components):
#         self.n_components = n_components
#         self.W = None
#
#     def fit(self, X):
#         if X.shape[0] < self.n_components:
#             raise ValueError("n_components is higher than height of X")
#
#         means = tf.reduce_mean(X, axis=0)
#         stds = tf.math.reduce_std(X, axis=0)
#         stds = tf.where(tf.equal(stds, 0), tf.ones_like(stds), stds)
#         X = (X - means) / stds
#
#         _, _, W = tf.linalg.svd(X)
#
#         self.W = W[:, : self.n_components]
#
#     def transform(self, X):
#         if self.W is None:
#             raise ValueError("Not fitted")
#         return tf.linalg.matmul(X, self.W)
#
#     def inverse_transform(self, X):
#         if self.W is None:
#             raise ValueError("Not fitted")
#         return tf.linalg.matmul(X, tf.transpose(self.W))
