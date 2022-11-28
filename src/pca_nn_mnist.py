import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA

tf.keras.utils.set_random_seed(1336)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 20

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
# pca = PCA(n_components=n_components, svd_solver="full", random_state=1336)

print("Fitting PCA")
start = time.time()
pca.fit(x_train_flat)

pca.transform(x_train_flat)
pca.transform(x_test_flat)


model = tf.keras.Sequential(
    [
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
print(f"Time taken: {time.time() - start}")
