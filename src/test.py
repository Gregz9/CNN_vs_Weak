import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.reshape(x_train, shape=[-1, 784])
x_test = tf.reshape(x_test, shape=[-1, 784])

# PCA implemented using tensors, to be able to run on gpu
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None

    def fit(self, X):
        if X.shape[0] < self.n_components:
            raise ValueError("n_components is higher than height of X")
        # assume design matrix

        # batch_size = train.shape[0]
        # features = train.shape[1] * train.shape[2]
        #
        # x_list = []
        # for i in range(batch_size):
        #     x_list.append(tf.reshape(train[i], (features,)))
        #
        # X = tf.stack(x_list)
        # X = tf.cast(X, dtype=tf.float32)

        means = tf.reduce_mean(X, axis=0)
        stds = tf.math.reduce_std(X, axis=0)
        stds = tf.where(tf.equal(stds, 0), tf.ones_like(stds), stds)
        X = (X - means) / stds

        _, _, W = tf.linalg.svd(X)

        self.W = W[:, : self.n_components]

    def transform(self, X):
        if self.W is None:
            raise ValueError("Not fitted")
        return tf.linalg.matmul(X, self.W)


pca = PCA(64)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


# class PCALayer(layers.Layer):
#     def __init__(self, num_outputs):
#         super(PCALayer, self).__init__()
#         self.num_outputs = num_outputs
#         self.trainable = False
#
#     def call(self, inputs):
#         batch_size = inputs.shape[0] or 1
#         features = inputs.shape[1] * inputs.shape[2]
#
#         x_list = []
#         for i in range(batch_size):
#             x_list.append(tf.reshape(inputs[i], (features,)))
#
#         X = tf.stack(x_list)
#
#         means = tf.reduce_mean(X)
#         print(means)
#
#         S, U, V = tf.linalg.svd(X)
#
#         slice_index = min(self.num_outputs, batch_size)
#         S = S[:slice_index]
#         S = tf.linalg.diag(S)
#         U = U[:, :slice_index]
#         output = tf.linalg.matmul(U, S)
#
#         return output

model = tf.keras.Sequential(
    [
        # layers.Rescaling(1.0 / 255),
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        # PCALayer(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6)
model.summary()
