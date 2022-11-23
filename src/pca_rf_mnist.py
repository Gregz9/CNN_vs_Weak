import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers
from PIL import Image
import time
import matplotlib.pyplot as plt
from utils import *
from sklearn.ensemble import RandomForestClassifier

batch_size = 128
epochs = 6
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_components = 100

x_train_flat = tf.reshape(x_train, shape=[-1, 784])
x_test_flat = tf.reshape(x_test, shape=[-1, 784])

W = PCA_fit(x_train_flat, n_components)

train_features = tf.matmul(x_train_flat[:10000], W)
test_features = tf.matmul(x_test_flat, W)

print(train_features.shape)
print(test_features.shape)

forest = RandomForestClassifier(random_state=1337)

forest.fit(train_features, y_train[:10000])

print(forest.score(test_features, y_test))
