import tensorflow as tf
from tensorflow.keras import layers


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