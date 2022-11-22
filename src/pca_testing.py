import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(y_train[:10])


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

    def inverse_transform(self, X):
        if self.W is None:
            raise ValueError("Not fitted")
        return tf.linalg.matmul(X, tf.transpose(self.W))


n_components = 100

flatmodel = tf.keras.Sequential(
    [
        # layers.Rescaling(1.0 / 255),
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        # PCALayer(64),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(10),
    ]
)

convmodel = tf.keras.Sequential(
    [
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

flatmodel.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

convmodel.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

start = time.time()
pca = PCA(n_components)
pca.fit(x_train_flat)
x_train_pca = pca.transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

flatmodel.fit(
    x_train_pca, y_train, validation_data=(x_test_pca, y_test), epochs=6, batch_size=64
)
print(f"PCA model time taken: {time.time() - start}")

x_train_4d = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test_4d = tf.reshape(x_test, shape=[-1, 28, 28, 1])

start = time.time()
convmodel.fit(
    x_train_4d, y_train, validation_data=(x_test_4d, y_test), epochs=6, batch_size=64
)
print(f"Conv model time taken: {time.time() - start}")

# ------- plotting pca ------------
x_train_remade = pca.inverse_transform(x_train_pca)

plt.subplot(421)
plt.title("Actual instance, \n 784 features")
plt.imshow(x_train[0])
plt.subplot(422)
plt.title(f"Constructed using only the \n {n_components} first principal components")
plt.imshow(tf.reshape(x_train_remade[0], (28, 28)))

plt.subplot(423)
plt.imshow(x_train[1])
plt.subplot(424)
plt.imshow(tf.reshape(x_train_remade[1], (28, 28)))

plt.subplot(425)
plt.imshow(x_train[2])
plt.subplot(426)
plt.imshow(tf.reshape(x_train_remade[2], (28, 28)))

plt.subplot(427)
plt.imshow(x_train[3])
plt.subplot(428)
plt.imshow(tf.reshape(x_train_remade[3], (28, 28)))

plt.show()

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
