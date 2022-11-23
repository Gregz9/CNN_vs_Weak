import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.ensemble import RandomForestClassifier

batch_size = 128
epochs = 1
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])

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

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output,
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
)

train_features = feature_extractor(x_train[:10000])
test_features = feature_extractor(x_test)

forest = RandomForestClassifier(random_state=1337)

forest.fit(train_features, y_train[:10000])

print(forest.score(test_features, y_test))
