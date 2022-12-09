import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
filedir = os.path.dirname(__file__)
import numpy as np
from CNN_models import DefaultConv2D, ResidualUnit
"""
Convolutional neural network using the ResNet-34 architecture presented in Gerons book on Machine Learning 
with Tensorflow used for classification of images contained in the pneumonia dataset.
"""
seed = 1337
tf.keras.utils.set_random_seed(seed)
# Forcing GPU to avoid running the calculations in arbitrary order
tf.config.experimental.enable_op_determinism()

TRAINDIR = filedir + "/../data/chest_xray/train"
TESTDIR = filedir + "/../data/chest_xray/test"
BATCHSIZE = 4 
IMG_HEIGHT = 227
IMG_WIDTH = 227

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINDIR,
    labels="inferred",
    seed=seed,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
    color_mode="grayscale",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TESTDIR,
    labels="inferred",
    seed=seed,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCHSIZE,
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

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0/255),
    DefaultConv2D(64, kernel_size=7, strides=2),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
    ])
prev_filters = 64 
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3: 
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.0003),
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003),
    # optimizer=tf.keras.optimizers.Adam(learning_rate=3*10e-6),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    batch_size=BATCHSIZE,
    validation_data=test_ds,
    epochs=10,
    class_weight=class_weight,
)
