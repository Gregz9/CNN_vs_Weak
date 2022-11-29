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
from sklearn.model_selection import KFold

tf.keras.utils.set_random_seed(1336)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 80

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
# pca = PCA(n_components=n_components, svd_solver="full", random_state=1336)

print("Fitting PCA")
start = time.time()
pca.fit(x_train_flat)

x_train_pca = pca.transform(x_train_flat)
x_train_pca = pca.transform(x_test_flat)

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
    directory="pca_nn_mnist-parameters",
)

tuner.search(x_train_pca, y_train, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
units = best_hps.get("units")
learning_rate = best_hps.get("learning_rate")

print(units)
print(learning_rate)
folds = 5

kf = KFold(n_splits=folds)  # random_state=1336)

avg_accuracy = 0
for train_index, test_index in kf.split(x_train_flat, y_train):
    train_start = train_index[0]
    train_stop = train_index[-1]

    test_start = test_index[0]
    test_stop = test_index[-1]

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)

    pca.fit(x_train_flat[train_start:train_stop])

    x_train_pca = pca.transform(x_train_flat[train_start:train_stop])
    x_holdout_pca = pca.transform(x_train_flat[test_start:test_stop])

    y_train_cv = y_train[train_start:train_stop]
    y_holdout = y_train[test_start:test_stop]

    model = tf.keras.Sequential(
        [
            layers.Dense(units, activation="relu"),
            layers.Dense(units, activation="relu"),
            layers.Dense(units, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        x_train_pca,
        y_train_cv,
        validation_data=(x_holdout_pca, y_holdout),
        epochs=6,
        batch_size=64,
    )

    results = model.evaluate(
        x_holdout_pca,
        y_holdout,
        return_dict=True,
    )

    print(results)
    avg_accuracy += results["accuracy"] / folds

print(avg_accuracy)
