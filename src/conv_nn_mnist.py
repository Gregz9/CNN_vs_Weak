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
import seaborn as sns

tf.keras.utils.set_random_seed(1336)
tf.config.experimental.enable_op_determinism()
"""
Convolutional neural network used for MNIST dataset. Performs hyperparameter tuning,
trains the model, prints average time for making a prediction on the entire test set
creates a confusion matrix
"""

batch_size = 128
epochs = 10
mnist = tf.keras.datasets.mnist

(x_train, y_train_integer), (x_test, y_test_integer) = mnist.load_data()

y_train = tf.one_hot(y_train_integer, 10)
y_test = tf.one_hot(y_test_integer, 10)

x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])


# Hyperparameter tuning model
def model_builder(hp):
    hp_width = hp.Choice("width", values=[3, 5, 7])

    model = tf.keras.Sequential(
        [
            layers.Rescaling(1.0 / 255),
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
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
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

# finding the best hyperparameters
tuner.search(x_train, y_train, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
width = best_hps.get("width")
learning_rate = best_hps.get("learning_rate")
print(f"Best kernel width found by tuner: {width}")
print(f"Best learning rate found by tuner: {learning_rate}")

# using the best hyperparameters
model = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, width, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(top_k=1),
        tf.keras.metrics.Recall(thresholds=0),
    ],
)

checkpoint_filepath = "/tmp/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

# fitting the model
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[model_checkpoint_callback],
)

# loading the best weights from the model
model.load_weights(checkpoint_filepath)

model.evaluate(x_test, y_test)

print("Timing model:")
with tf.device("/cpu:0"):
    timeit(model.predict, (x_test, y_test), batch_size=batch_size)

predictions = tf.math.argmax(model.predict(x_test), axis=1)
conf = conf_mat(predictions, y_test_integer, num_cls=10)
conf = perc(conf)

plot_confusion(conf, title="Confusion matrix - CNN - MNIST")
print(results)
