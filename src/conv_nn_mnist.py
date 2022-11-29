import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import KFold
import keras_tuner as kt

tf.keras.utils.set_random_seed(1336)
batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])

# print(x_train.shape)
#
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# print(train_ds)
#
# train_ds = train_ds.batch(batch_size)
# val_ds = val_ds.batch(batch_size)


def model_builder(hp):
    hp_width = hp.Choice("width", values=[3, 5, 7])

    model = tf.keras.Sequential(
        [
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, hp_width, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, hp_width, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, hp_width, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
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
    directory="conv_nn_mnist-params",
)

tuner.search(x_train, y_train, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
width = best_hps.get("width")
learning_rate = best_hps.get("learning_rate")
print(width)
print(learning_rate)

kf = KFold(n_splits=5)  # random_state=1336)


avg_accuracy = 0
for train_index, test_index in kf.split(x_train, y_train):
    train_start = train_index[0]
    train_stop = train_index[-1]

    test_start = test_index[0]
    test_stop = test_index[-1]

    model = tf.keras.Sequential(
        [
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, width, activation="relu", padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, width, activation="relu", padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, width, activation="relu", padding='same'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        x_train[train_start:train_stop],
        y_train[train_start:train_stop],
        validation_data=(x_train[test_start:test_stop], y_train[test_start:test_stop]),
        epochs=epochs,
    )

    results = model.evaluate(
        x_train[test_start:test_stop],
        y_train[test_start:test_stop],
        return_dict=True,
    )

    print(results)
    avg_accuracy += results["accuracy"] / 5

print(avg_accuracy)
