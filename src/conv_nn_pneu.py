import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import keras_tuner as kt
from utils import *

"""
This file contains code for a CNN model using AlexNet architecture ( 
won the ILSVRC challenge in 2012) which is applied to the Pneumonia dataset.
The script outputs the final training and test accuracies, and plots 
a confusion matrix after the model has finished prediction.
"""


filedir = os.path.dirname(__file__)
from functools import partial

seed = 1336
tf.keras.utils.set_random_seed(seed)
# Module from tensorflow enabling override of GPU settings
tf.config.experimental.enable_op_determinism()

# ------------------------------- Loading data ------------------------------------
TRAINDIR = filedir + "/../data/chest_xray/train"
TESTDIR = filedir + "/../data/chest_xray/test"
BATCHSIZE = 256
IMG_HEIGHT = 227
IMG_WIDTH = 227

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINDIR,
    validation_split=0.1,
    subset="training",
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINDIR,
    validation_split=0.1,
    subset="validation",
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
# --------------------------- Adjusting class ratio ----------------------------
COUNT_NORMAL = 1071
COUNT_PNEUMONIA = 3114

weight_for_0 = (1 / COUNT_NORMAL) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------ Hyperparameter tuner ---------------------------------
def model_builder(hp):
    hp_learning_rate = hp.Choice(
        "learning_rate", values=[2 * 1e-5, 2 * 1e-4, 2 * 1e-3, 2 * 1e-2, 2 * 1e-1]
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(
                96,
                11,
                strides=(4, 4),
                padding="valid",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"),
            tf.keras.layers.Conv2D(
                256,
                5,
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(3, 3), strides=2, padding="valid"
            ),  # S4
            tf.keras.layers.Conv2D(
                384,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),  # C5
            tf.keras.layers.Conv2D(
                384,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),  # C6
            tf.keras.layers.Conv2D(
                256,
                3,
                strides=(1, 1),
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                4096,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                4096,
                activation="relu",
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


# --------------------------------- AlexNet CNN ----------------------------------
learning_rate = 0.001
model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(
            96,
            11,
            strides=(4, 4),
            padding="valid",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"),
        tf.keras.layers.Conv2D(
            256,
            5,
            strides=(1, 1),
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=2, padding="valid"
        ),  # S4
        tf.keras.layers.Conv2D(
            384,
            3,
            strides=(1, 1),
            padding="same",
            activation="relu",
        ),  # C5
        tf.keras.layers.Conv2D(
            384,
            3,
            strides=(1, 1),
            padding="same",
            activation="relu",
        ),  # C6
        tf.keras.layers.Conv2D(
            256,
            3,
            strides=(1, 1),
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            4096,
            activation="relu",
        ),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            4096,
            activation="relu",
        ),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2 * 10e-6),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(thresholds=0),
        tf.keras.metrics.Recall(thresholds=0),
    ],
)
# Saves the weights from the best trainig epoch
checkpoint_filepath = "/tmp/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

model.fit(
    train_ds,
    batch_size=BATCHSIZE,
    validation_data=test_ds,
    epochs=7,
    class_weight=class_weight,
    callbacks=[model_checkpoint_callback],
)

model.load_weights(checkpoint_filepath)

model.evaluate(test_ds, batch_size=BATCHSIZE)

# Average prediction time measurement
print("Timing model:")
timeit(model.predict, test_ds, batch_size=BATCHSIZE)

# making confusion matrix
y_test = None
for _, labels in test_ds:
    if y_test is None:
        y_test = labels

    else:
        y_test = tf.concat([y_test, labels], axis=0)

# -------------------------- Plotting the confusion matrix -----------------------------
pred = model.predict(test_ds)
pred = tf.where(pred > 0, 1, 0)
pred = tf.squeeze(pred)
conf = conf_mat(pred, y_test, 2)
conf = perc(conf)

plot_confusion(conf, title="Confusion matrix - CNN - PNEUMONIA")
