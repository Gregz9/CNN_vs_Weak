import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)
"""
Convolutional neural network used for pneumonia dataset. Builds and fits data.
"""

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

COUNT_NORMAL = 1071
COUNT_PNEUMONIA = 3114

weight_for_0 = (1 / COUNT_NORMAL) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (COUNT_NORMAL + COUNT_PNEUMONIA) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

kernel = tf.keras.regularizers.L2(l2=0.001)
bias = tf.keras.regularizers.L2(l2=0.001)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(
            96, 11, strides=(4,4), padding='valid', activation="relu"), #kernel_regularizer=kernel, bias_regularizer=bias),

        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'),
        tf.keras.layers.Conv2D(
            256, 5, strides=(1,1), padding='same', activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias), 
          
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'), # S4
        tf.keras.layers.Conv2D(
            384, 3, strides=(1,1), padding='same', activation="relu"), #kernel_regularizer=kernel, bias_regularizer=bias), #C5
        tf.keras.layers.Conv2D(
            384, 3, strides=(1,1), padding='same', activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias), #C6
        tf.keras.layers.Conv2D(
            256, 3, strides=(1,1), padding='same', activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias), 
        
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'), 
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            4096, activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias
        #),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            4096, activation="relu"), #, kernel_regularizer=kernel, bias_regularizer=bias
        #),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer="adamax",
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
