import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA

filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)
"""
PCA neural network used for pneumonia dataset. Builds and fits data, takes time.
"""

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
X_train = None
X_test = None
y_train = None
y_test = None

n_components = 5000
kernel = tf.keras.regularizers.L2(l2=0.001)
bias = tf.keras.regularizers.L2(l2=0.001)
model = tf.keras.Sequential(
    [
        layers.Dense(
            n_components,
            kernel_regularizer=kernel,
            bias_regularizer=bias,
            activation="relu",
        ),
        layers.Dense(
            n_components,
            kernel_regularizer=kernel,
            bias_regularizer=bias,
            activation="relu",
        ),
        layers.Dense(
            n_components,
            kernel_regularizer=kernel,
            bias_regularizer=bias,
            activation="relu",
        ),
        layers.Dense(1),
    ]
)

for batch, labels in train_ds:
    if X_train is None:
        X_train = tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0
        y_train = labels

    else:
        X_train = tf.concat(
            [X_train, (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0)], axis=0
        )
        y_train = tf.concat([y_train, labels], axis=0)

for batch, labels in val_ds:
    if X_test is None:
        X_test = tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0
        y_test = labels

    else:
        X_test = tf.concat(
            [X_test, (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0)], axis=0
        )
        y_test = tf.concat([y_test, labels], axis=0)


pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
# pca = PCA(n_components=n_components, svd_solver="full", random_state=1336)

print("Fitting PCA")
start = time.time()
pca.fit(X_train)

pca.transform(X_train)
pca.transform(X_test)


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=6,
    batch_size=64,
)
print(f"Time taken: {time.time() - start}")

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
