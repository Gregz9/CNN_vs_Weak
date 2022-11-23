import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *

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
    batch_size=128,
    color_mode="grayscale",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TESTDIR,
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=128,
    color_mode="grayscale",
)


# grab a subset for PCA calculation
x_list = []
i = 0
for batch, _ in train_ds:
    x_list.append(tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0)
    i += 1
    if i > 20:
        break

X_subset = tf.concat(x_list, axis=0)
n_components = 1000

W = PCA_fit(X_subset, n_components)


model = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        PCALayer(W),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(n_components, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=6, batch_size=64)


# ------- plotting pca ------------
# x_train_remade = pca.inverse_transform(x_train_pca)

# plt.subplot(421)
# plt.title("Actual instance, \n 784 features")
# plt.imshow(x_train[0])
# plt.subplot(422)
# plt.title(f"Constructed using only the \n {n_components} first principal components")
# plt.imshow(tf.reshape(x_train_remade[0], (28, 28)))
#
# plt.subplot(423)
# plt.imshow(x_train[1])
# plt.subplot(424)
# plt.imshow(tf.reshape(x_train_remade[1], (28, 28)))
#
# plt.subplot(425)
# plt.imshow(x_train[2])
# plt.subplot(426)
# plt.imshow(tf.reshape(x_train_remade[2], (28, 28)))
#
# plt.subplot(427)
# plt.imshow(x_train[3])
# plt.subplot(428)
# plt.imshow(tf.reshape(x_train_remade[3], (28, 28)))
#
# plt.show()
