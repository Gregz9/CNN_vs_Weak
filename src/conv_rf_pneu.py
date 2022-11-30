import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
import tensorflow_decision_forests as tfds

tf.keras.utils.set_random_seed(1336)
"""
Convolutional random forest used for pneumonia dataset. Builds and fits data.
"""

batch_size = 128
epochs = 3
mnist = tf.keras.datasets.mnist

filedir = os.path.dirname(__file__)

TRAINDIR = filedir + "/../data/chest_xray/train"
TESTDIR = filedir + "/../data/chest_xray/test"
BATCHSIZE = 128
IMG_HEIGHT = 200
IMG_WIDTH = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINDIR,
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TESTDIR,
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(train_ds)


model = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1),
    ]
)


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=epochs)
# model.fit(val_ds, epochs=epochs)

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output,
)

features_train = train_ds.map(lambda batch, label: (feature_extractor(batch), label))
features_test = val_ds.map(lambda batch, label: (feature_extractor(batch), label))
print(features_test)
print(features_train)


forest = tfds.keras.RandomForestModel(
    verbose=1, max_depth=16, random_seed=1337, check_dataset=False
)
forest.fit(x=features_train)

forest.compile(metrics=["accuracy"])

print(forest.evaluate(features_train, return_dict=True))
print(forest.evaluate(features_test, return_dict=True))
