import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA
import keras_tuner as kt

tf.keras.utils.set_random_seed(1336)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 20


def model_builder(hp):
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model = tf.keras.Sequential(
        [
            layers.Dense(units=hp_units, activation="relu"),
            layers.Dense(units=hp_units, activation="relu"),
            layers.Dense(units=hp_units, activation="relu"),
            layers.Dense(10),
        ]
    )

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


tuner = kt.Hyperband(
    model_builder,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
)


x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
# pca = PCA(n_components=n_components, svd_solver="full", random_state=1336)

print("Fitting PCA")
start = time.time()
pca.fit(x_train_flat)

pca.transform(x_train_flat)
pca.transform(x_test_flat)

tuner.search(x_train_flat, y_train, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.get("units"))
print(best_hps.get("learning_rate"))

model.fit(
    x_train_flat,
    y_train,
    validation_data=(x_test_flat, y_test),
    epochs=6,
    batch_size=64,
)
print(f"Time taken: {time.time() - start}")
