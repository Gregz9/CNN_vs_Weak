import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import keras_tuner as kt
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA
import tensorflow_decision_forests as tfdf

filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)
"""
PCA neural network used for pneumonia dataset. Builds and fits data, takes time.
"""


def model_builder(hp):
    hp_lambda = hp.Choice("lambda", values=[1e-5, 1e-4, 1e-3])
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-2, 1e-1])

    kernel = tf.keras.regularizers.L2(l2=hp_lambda)
    bias = tf.keras.regularizers.L2(l2=hp_lambda)

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

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


with tf.device("/cpu:0"):
    TRAINDIR = filedir + "/../data/chest_xray/train"
    TESTDIR = filedir + "/../data/chest_xray/test"
    batch_size = 128
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAINDIR,
        labels="inferred",
        seed=1337,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        color_mode="grayscale",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TESTDIR,
        labels="inferred",
        seed=1337,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        color_mode="grayscale",
    )

    COUNT_NORMAL = 1071
    COUNT_PNEUMONIA = 3114

    weight_for_0 = (1 / COUNT_NORMAL) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0
    weight_for_1 = (1 / COUNT_PNEUMONIA) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    x_list = []
    i = 0
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    n_components = 9

    for batch, labels in train_ds:
        if X_train is None:
            X_train = tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0
            y_train = labels

        else:
            X_train = tf.concat(
                [X_train, (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0)], axis=0
            )
            y_train = tf.concat([y_train, labels], axis=0)

    for batch, labels in test_ds:
        if X_test is None:
            X_test = tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0
            y_test = labels

        else:
            X_test = tf.concat(
                [X_test, (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0)], axis=0
            )
            y_test = tf.concat([y_test, labels], axis=0)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)

    print("Fitting PCA")
    start = time.time()
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    forest = tfdf.keras.RandomForestModel(
        verbose=1, max_depth=40, random_seed=1337, check_dataset=False
    )
    forest.fit(X_train_pca, y_train, class_weight=class_weight)

    forest.compile(metrics=["accuracy"])

    print(forest.evaluate(X_train_pca, y_train, return_dict=True))
    print(forest.evaluate(X_test_pca, y_test, return_dict=True))

    def predict():
        X_test_pca = pca.transform(X_test)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_pca, y_test))
        test_ds = test_ds.batch(batch_size)
        forest.predict(test_ds)

    print("Timing prediction")
    timeit(predict)
