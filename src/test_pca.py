import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import PCA

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_flat = x_train.reshape((x_train.shape[0], 784))
x_test_flat = x_test.reshape((x_test.shape[0], 784))
x_train_flat = x_train_flat


pca35 = PCA(n_components=35, svd_solver="randomized")
pca50 = PCA(n_components=50, svd_solver="randomized")
pca100 = PCA(n_components=100, svd_solver="randomized")

pca35.fit(x_train_flat)
temp = pca35.transform(x_train_flat)
x35 = pca35.inverse_transform(temp)

pca50.fit(x_train_flat)
temp = pca50.transform(x_train_flat)
x50 = pca50.inverse_transform(temp)

pca100.fit(x_train_flat)
temp = pca100.transform(x_train_flat)
x100 = pca100.inverse_transform(temp)


# ------- plotting pca ------------
plt.subplot(431)
plt.title("Actual instance, \n 784 features")
plt.imshow(tf.reshape(x_train[0], (28, 28)))
plt.subplot(432)
plt.title(f"Constructed using only the \n {n_components} first principal components")
plt.imshow(tf.reshape(x35[0], (28, 28)))
plt.subplot(433)
plt.title(
    f"Constructed using only the \n {n_components} first principal components but its stochastic"
)
plt.imshow(tf.reshape(x50[0], (28, 28)))

plt.subplot(434)
plt.imshow(tf.reshape(x_train[1], (28, 28)))
plt.subplot(435)
plt.imshow(tf.reshape(x35[1], (28, 28)))
plt.subplot(436)
plt.imshow(tf.reshape(x50[1], (28, 28)))

plt.subplot(437)
plt.imshow(tf.reshape(x_train[2], (28, 28)))
plt.subplot(438)
plt.imshow(tf.reshape(x35[2], (28, 28)))
plt.subplot(439)
plt.imshow(tf.reshape(x50[2], (28, 28)))

plt.show()
