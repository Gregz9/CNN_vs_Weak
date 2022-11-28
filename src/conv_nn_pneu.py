import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers

filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)

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

test_ds = tf.keras.utils.image_dataset_from_directory(
    TESTDIR,
    labels="inferred",
    seed=1337,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

kernel = tf.keras.regularizers.L2(l2=0.001)
bias = tf.keras.regularizers.L2(l2=0.001)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 7, activation="relu", kernel_regularizer=kernel, bias_regularizer=bias),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 7, activation="relu", kernel_regularizer=kernel, bias_regularizer=bias),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 7, activation="relu", kernel_regularizer=kernel, bias_regularizer=bias),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=kernel, bias_regularizer=bias),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=test_ds, epochs=6)
