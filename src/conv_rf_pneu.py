import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
import keras_tuner as kt
import tensorflow_decision_forests as tfdf
from utils import *

"""
This script contains an implementation of a CNN model constructed using the 
tensorflow API and utlizing the AlexNet architecture. This model is first applied 
to the Pneumonia dataset before its output is fed into a Random Decision Forest 
as an attempt to improve upon the results achieved solely by the CNN architecture 
used here. 
"""

filedir = os.path.dirname(__file__)
from functools import partial

seed = 1336
tf.keras.utils.set_random_seed(seed)
# Module from tensorflow forcing GPU to run deterministically
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


tuner = kt.Hyperband(
    model_builder,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="conv_nn_pneu-params",
)

tuner.search(train_ds, epochs=10, validation_data=val_ds, class_weight=class_weight)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
learning_rate = best_hps.get("learning_rate")

# --------------------------------- AlexNet CNN ----------------------------------
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=2 * 10e-6),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
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
    train_ds,
    batch_size=BATCHSIZE,
    validation_data=test_ds,
    epochs=7,
    class_weight=class_weight,
    callbacks=[model_checkpoint_callback],
)

model.load_weights(checkpoint_filepath)

model.evaluate(test_ds, batch_size=BATCHSIZE)

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-1].output,
)
# Hyperparameter tuning for the Random Forest
features_train = train_ds.map(lambda batch, label: (feature_extractor(batch), label))
features_test = test_ds.map(lambda batch, label: (feature_extractor(batch), label))
print(features_test)
print(features_train)

tuner = tfdf.tuner.RandomSearch(num_trials=20)
tuner.choice("max_depth", [5, 10, 15, 20, 25, 30])
tuner.choice("min_examples", [5, 7, 9, 11, 13])

# ------------------------------ Random Forest model ------------------------------------
# forest = tfdf.keras.RandomForestModel(max_depth=3, min_examples=9, check_dataset=False)
forest = tfdf.keras.RandomForestModel(tuner=tuner, check_dataset=False)
forest.fit(x=features_train, class_weight=class_weight)

forest.compile(metrics=["accuracy"])

print(forest.evaluate(features_train, return_dict=True))
print(forest.evaluate(features_test, return_dict=True))


def predict():
    features_test = test_ds.map(lambda batch, label: (feature_extractor(batch), label))
    forest.predict(features_test)


print("Timing prediction")
timeit(predict)
