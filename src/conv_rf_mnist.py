import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
import tensorflow_decision_forests as tfdf

tf.keras.utils.set_random_seed(1336)
"""
Convolutional random forest used for MNIST dataset. Builds and fits data.
"""

batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

width = 5
learning_rate = 0.001

model = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, width, activation="relu"),
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
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    # batch_size=batch_size,
)

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output,
)

features_train = train_ds.map(lambda batch, label: (feature_extractor(batch), label))
features_test = test_ds.map(lambda batch, label: (feature_extractor(batch), label))

forest = tfdf.keras.RandomForestModel(
    verbose=1,
    # max_depth=25,
    random_seed=1337,
    # num_trees=300,  # , tuner=tuner#, check_dataset=False
)
forest.fit(x=features_train)

forest.compile(metrics=["accuracy"])

print(forest.evaluate(features_train, return_dict=True))
print(forest.evaluate(features_test, return_dict=True))
