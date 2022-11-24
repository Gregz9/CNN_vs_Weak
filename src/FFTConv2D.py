import tensorflow as tf 
import numpy as np 
from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
import time 
import matplotlib.pyplot as plt 
import os
import imageio

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
# DFT AND FFT should be more efficient when the image size is a number of power of 2
org_size = x_train[0].shape
kernel_size = 3 
new_size = 2**int(np.ceil(np.log2(x_train.shape[-1] + 2*(kernel_size//2)))) 
total_pad = new_size - x_train.shape[-1]

filter_ = tf.ones(shape=(3,3))/(3**2)

image = tf.signal.rfft2d(x_train[0], fft_length=x_train[0].shape)
kernel = tf.signal.rfft2d(filter_, fft_length=x_train[0].shape)
print(f'{image.shape=}')
print(f'{kernel.shape=}')

fft_image = image*kernel
ifft_image = tf.signal.irfft2d(fft_image, fft_length=tf.shape(x_train[0]))

fig, ax = plt.subplots(1,2)

ax[0].imshow(ifft_image, cmap='gray', vmin=0, vmax=255,aspect='auto')
ax[0].title.set_text('fft image')
ax[1].imshow(x_train[0], cmap='gray', vmin=0, vmax=255,aspect='auto')
ax[1].title.set_text('original image')
plt.show()
"""

class FFTConv2d(tf.keras.layers.Conv2D):

    def convolution_op(self, inputs, kernel):
        print(f'{inputs.shape=}')
        print(f'{kernel.shape=}')
        
        image = tf.signal.rfft2d(inputs, fft_length=inputs.shape)
        kernel = tf.signal.rfft2d(kernel, fft_length=inputs.shape)
        print(f'{image.shape=}')
        print(f'{kernel.shape=}')
        
model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        FFTConv2d(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
x_train_4d = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test_4d = tf.reshape(x_test, shape=[-1, 28, 28, 1])

model.fit(
    x_train_4d, y_train, validation_data=(x_test_4d, y_test), epochs=6, 
)

