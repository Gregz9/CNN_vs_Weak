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

print(x_train.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(train_ds)

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

model = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)
