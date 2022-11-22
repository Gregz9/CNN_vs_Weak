import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.reshape(x_train, shape=[-1, 784])
x_test = tf.reshape(x_test, shape=[-1, 784])

filedir = os.path.dirname(__file__)

TRAINDIR = filedir + "/../data/chest_xray/train"
TESTDIR = filedir + "/../data/chest_xray/test"
BATCHSIZE = 64
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

test_ds = tf.keras.utils.image_dataset_from_directory(
    TESTDIR,
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)

normalization_layer = layers.Rescaling(1.0 / 255)

AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


def PCAB(train, test, n_components):
    batch_size = train.shape[0]
    features = train.shape[1] * train.shape[2]

    x_list = []
    for i in range(batch_size):
        x_list.append(tf.reshape(train[i], (features,)))

    X = tf.stack(x_list)
    X = tf.cast(X, dtype=tf.float32)

    means = tf.reduce_mean(X, axis=0)
    X = X - means

    S, U, V = tf.linalg.svd(X)

    slice_index = min(n_components, batch_size)
    S = S[:slice_index]
    S = tf.linalg.diag(S)
    U = U[:, :slice_index]
    output = tf.linalg.matmul(U, S)
    return output


pca = PCA(n_components=64)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


class PCALayer(layers.Layer):
    def __init__(self, num_outputs):
        super(PCALayer, self).__init__()
        self.num_outputs = num_outputs
        self.trainable = False

    def call(self, inputs):
        batch_size = inputs.shape[0] or 1
        features = inputs.shape[1] * inputs.shape[2]

        x_list = []
        for i in range(batch_size):
            x_list.append(tf.reshape(inputs[i], (features,)))

        X = tf.stack(x_list)

        means = tf.reduce_mean(X)
        print(means)

        S, U, V = tf.linalg.svd(X)

        slice_index = min(self.num_outputs, batch_size)
        S = S[:slice_index]
        S = tf.linalg.diag(S)
        U = U[:, :slice_index]
        output = tf.linalg.matmul(U, S)

        return output


model = tf.keras.Sequential(
    [
        # layers.Rescaling(1.0 / 255),
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        # PCALayer(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)


# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# model.build((64, 200, 200, 1))
# model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6)

# model.fit(train_ds, validation_data=test_ds, epochs=6)
