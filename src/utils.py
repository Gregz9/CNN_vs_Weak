import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def PCA_stoch(X, n_components):
    print("eric")


def PCA_stoch(X, n_components, iterations=10, eta=1e-5, convergence=1e-5, epochs=10):
    n, d = X.shape
    W_t = np.random.rand(d, n_components) - 0.5
    print(W_t.shape)
    W_t = W_t / np.linalg.norm(W_t, axis=0)
    X = X - np.mean(X, axis=0)

    for e in range(epochs):

        W = W_t

        for t in range(iterations):
            i = np.random.randint(n)
            print(f"{W.shape=}")
            print(f"{X[i].shape=}")
            print(f"{X[i].T.dot(W).shape=}")
            W_mark = W + eta * (X[i] * (X[i].T.dot(W) - X[i].T.dot(W_t)))
            W_mark = W_mark / np.linalg.norm(W_mark, axis=0)
            W = _W

        d = np.linalg.norm(W_t - W)
        W_t = W

        if d < rate:
            return


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
