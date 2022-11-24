import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *

batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
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
