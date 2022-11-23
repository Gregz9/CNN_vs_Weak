import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
import tensorflow_decision_forests as tfds

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
    if i > 40:
        break

X_subset = tf.concat(x_list, axis=0)
n_components = 2000

W = PCA_fit(X_subset, n_components)

print(train_ds)
train_ds = train_ds.map(
    lambda batch, label: (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0, label)
)
val_ds = val_ds.map(
    lambda batch, label: (tf.reshape(batch, shape=[-1, 200 * 200]) / 255.0, label)
)
print(train_ds)
train_ds = train_ds.map(lambda batch, label: (tf.matmul(batch, W), label))
val_ds = val_ds.map(lambda batch, label: (tf.matmul(batch, W), label))
print(train_ds)
print(val_ds)


model = tfds.keras.RandomForestModel(verbose=1, max_depth=10, check_dataset=False)

model.fit(x=train_ds)
model.compile(metrics=["accuracy"])
print(model.evaluate(train_ds, return_dict=True))
print(model.evaluate(val_ds, return_dict=True))
