import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
import tensorflow_decision_forests as tfdf
from sklearn.decomposition import PCA

tf.keras.utils.set_random_seed(1336)
"""
PCA random forest used for MNIST dataset. Builds PCA Random Forest hybrid model using
SciKit Learns randomized PCA and Tensorflow decision forest libraries. Trains and tests
model.
"""

batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# components to use after PCA
n_components = 50

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

# define and fit PCA
pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
pca.fit(x_train_flat)

x_train_pca = pca.transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_pca, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test_pca, y_test))

train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

# tune model
tuner = tfdf.tuner.RandomSearch(num_trials=20)
tuner.choice("max_depth", [5, 10, 15, 20, 25, 30])
tuner.choice("min_examples", [5, 7, 9, 11, 13])

# random forest
forest = tfdf.keras.RandomForestModel(random_seed=1336, tuner=tuner, check_dataset=False)
forest.fit(x=train_ds)

forest.compile(metrics=["accuracy"])

print(forest.evaluate(train_ds, return_dict=True))
print(forest.evaluate(test_ds, return_dict=True))


def predict():
    x_test_pca = pca.transform(x_test_flat)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test_pca, y_test))
    test_ds = test_ds.batch(batch_size)
    forest.predict(test_ds)

# time CPU runtime
with tf.device("/cpu:0"):
    print("Timing prediction")
    timeit(predict)
