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

batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 100

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

start = time.time()

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=1336)
pca.fit(x_train_flat)

x_train_pca = pca.transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_pca, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_test_pca, y_test))

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)


forest = tfdf.keras.RandomForestModel(
    verbose=1,
    max_depth=16,
    random_seed=1336,
)

forest.fit(x=train_ds)

forest.compile(metrics=["accuracy"])

print(forest.evaluate(train_ds, return_dict=True))
print(forest.evaluate(test_ds, return_dict=True))
