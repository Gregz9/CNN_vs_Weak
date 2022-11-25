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

# train_ds.map(lambda batch, label: (tf.matmul(batch, W), label))

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
