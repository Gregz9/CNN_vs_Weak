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

"""
This script contains an implementation of a neural network constructed by using the 
tensorflow API which is then fed the first n principal components extracted through the 
use of the SciKit-learns randomized PCA implementation from X-Ray images of Pneumonia obesrved in children.
The sript outputs the training and test accurcies achieved, and average time spent on prediction (time 
spent on training exluded). 
"""

filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)
# Module from tensorflow enabling override of GPU settings
tf.config.experimental.enable_op_determinism()

# ------------------------------------ Hyperparameter tuner  ------------------------------------
def model_builder(hp):
    hp_lambda = hp.Choice("lambda", values=[1e-5, 1e-4, 1e-3])
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    hp_units = hp.Choice("units", values=[9, 15, 17, 24, 30])

    kernel = tf.keras.regularizers.L2(l2=hp_lambda)
    bias = tf.keras.regularizers.L2(l2=hp_lambda)

    model = tf.keras.Sequential(
        [
            layers.Dense(
                hp_units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(
                hp_units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(
                hp_units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


# ------------------------------------- Loading data -------------------------------------------
with tf.device("/cpu:0"):
    TRAINDIR = filedir + "/../data/chest_xray/train"
    TESTDIR = filedir + "/../data/chest_xray/test"
    BATCHSIZE = 256
    IMG_HEIGHT = 227
    IMG_WIDTH = 227

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

    # ----------------------------------- Preprocessing ---------------------------------------------
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
            X_train = tf.reshape(batch, shape=[-1, IMG_WIDTH * IMG_HEIGHT]) / 255.0
            y_train = labels

        else:
            X_train = tf.concat(
                [
                    X_train,
                    (tf.reshape(batch, shape=[-1, IMG_HEIGHT * IMG_WIDTH]) / 255.0),
                ],
                axis=0,
            )
            y_train = tf.concat([y_train, labels], axis=0)

    for batch, labels in test_ds:
        if X_test is None:
            X_test = tf.reshape(batch, shape=[-1, IMG_HEIGHT * IMG_WIDTH]) / 255.0
            y_test = labels

        else:
            X_test = tf.concat(
                [
                    X_test,
                    (tf.reshape(batch, shape=[-1, IMG_HEIGHT * IMG_WIDTH]) / 255.0),
                ],
                axis=0,
            )
            y_test = tf.concat([y_test, labels], axis=0)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)

    print("Fitting PCA")
    start = time.time()
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

# ------------------------------- Hyperparamter tuning -------------------------------------------
with tf.device("/gpu:0"):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="pca_nn_pneu-params",
    )

    tuner.search(
        X_train_pca, y_train, epochs=20, validation_split=0.2, class_weight=class_weight
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    lam = best_hps.get("lambda")
    learning_rate = best_hps.get("learning_rate")
    units = best_hps.get("units")

    print(f"{lam=}")
    print(f"{learning_rate=}")
    print(f"{units=}")

    # --------------------------------- Neural Network model ----------------------------------------
    kernel = tf.keras.regularizers.L2(l2=lam)
    bias = tf.keras.regularizers.L2(l2=lam)
    model = tf.keras.Sequential(
        [
            layers.Dense(
                units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(
                units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(
                units,
                kernel_regularizer=kernel,
                bias_regularizer=bias,
                activation="relu",
            ),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(thresholds=0),
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

    model.fit(
        X_train_pca,
        y_train,
        validation_data=(X_test_pca, y_test),
        epochs=15,
        batch_size=BATCHSIZE,  # 64,
        class_weight=class_weight,
        callbacks=[model_checkpoint_callback],
    )
    model.load_weights(checkpoint_filepath)

    model.evaluate(X_test_pca, y_test, batch_size=BATCHSIZE)

    def predict():
        x_test_pca = pca.transform(X_test)
        model.predict((x_test_pca, y_test))

    # time gpu
    print("Timing prediction GPU")
    timeit(predict)

# time cpu
with tf.device("/cpu:0"):
    print("Timing prediction CPU")
    timeit(predict)

    X_train_pca_restored = pca.inverse_transform(X_test_pca)
    # ------------------------ Images reconstructed using only n principal components -----------------------------
    plt.subplot(421)
    plt.title(f"Actual instance, 51k features. Class {y_train[0]}", size=18)
    plt.imshow(tf.reshape(X_train[0], (227, 227)))
    plt.subplot(422)
    plt.title("PCA with 9 components", size=18)
    plt.imshow(tf.reshape(X_train_pca_restored[0], (227, 227)))

    plt.subplot(423)
    plt.title(f"Class {y_train[5]}", size=18)
    plt.imshow(tf.reshape(X_train[5], (227, 227)))
    plt.subplot(424)
    plt.imshow(tf.reshape(X_train_pca_restored[5], (227, 227)))

    plt.subplot(425)
    plt.title(f"Class {y_train[8]}", size=18)
    plt.imshow(tf.reshape(X_train[8], (227, 227)))
    plt.subplot(426)
    plt.imshow(tf.reshape(X_train_pca_restored[8], (227, 227)))

    plt.subplot(427)
    plt.title(f"Class {y_train[29]}", size=18)
    plt.imshow(tf.reshape(X_train[29], (227, 227)))
    plt.subplot(428)
    plt.imshow(tf.reshape(X_train_pca_restored[29], (227, 227)))

    plt.show()

    pred = model.predict((X_test_pca, y_test))
    pred = tf.where(pred > 0, 1, 0)
    pred = tf.squeeze(pred)
    conf = conf_mat(pred, y_test, 2)
    conf = perc(conf)

    plot_confusion(conf, title="Confusion matrix - PCA_NN - PNEUMONIA")
