import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 100

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

W = PCA_fit(x_train_flat, n_components)

model = tf.keras.Sequential(
    [
        PCALayer(W),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    x_train_flat,
    y_train,
    validation_data=(x_test_flat, y_test),
    epochs=6,
    batch_size=64,
)

# ------- plotting pca ------------
# x_train_remade = pca.inverse_transform(x_train_pca)
#
# plt.subplot(421)
# plt.title("Actual instance, \n 784 features")
# plt.imshow(x_train[0])
# plt.subplot(422)
# plt.title(f"Constructed using only the \n {n_components} first principal components")
# plt.imshow(tf.reshape(x_train_remade[0], (28, 28)))
#
# plt.subplot(423)
# plt.imshow(x_train[1])
# plt.subplot(424)
# plt.imshow(tf.reshape(x_train_remade[1], (28, 28)))
#
# plt.subplot(425)
# plt.imshow(x_train[2])
# plt.subplot(426)
# plt.imshow(tf.reshape(x_train_remade[2], (28, 28)))
#
# plt.subplot(427)
# plt.imshow(x_train[3])
# plt.subplot(428)
# plt.imshow(tf.reshape(x_train_remade[3], (28, 28)))
#
# plt.show()

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
