import tensorflow as tf 
import numpy as np 
from tensorflow.keras import layers
import time 
import matplotlib.pyplot as plt 
import os
import imageio

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
print(x_train.shape)
# DFT AND FFT should be more efficient when the image size is a number of power of 2
org_size = x_train[0].shape
kernel_size = 3 
new_size = 2**int(np.ceil(np.log2(x_train.shape[-1] + 2*(kernel_size//2)))) 

total_pad = new_size - x_train.shape[-1]

filter_ = tf.ones(shape=(5,5))/(5**2)

image = tf.signal.rfft2d(x_train[0], fft_length=[new_size, new_size])
kernel = tf.signal.rfft2d(filter_, fft_length=[new_size, new_size])

fft_image = image*kernel
ifft_image = tf.signal.irfft2d(fft_image, fft_length=[new_size, new_size])

fig, ax = plt.subplots(1,2)

ax[0].imshow(ifft_image, cmap='gray', vmin=0, vmax=255,aspect='auto')
ax[0].title.set_text('fft image')
ax[1].imshow(x_train[0], cmap='gray', vmin=0, vmax=255,aspect='auto')
ax[1].title.set_text('original image')
plt.show()
"""
class FFTConv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, strides, name='fftconv2d'):
        super().__init__(name=name)
        self.filters=filters
        self.kernel_size
        self.strides = strides 
        self.padding = padding

    def build(self, input_shape):

        kernel_shape = self.kernel_size + (input_shape[-1] //self.groups, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer='random_normal', trainable=True)           
        

    def call(self, inputs):
        newSize = 2**int(np.ceil(np.log2(inputs.shape[2] + 2*(self.kernel_size[0]//2))))
        final_padding = int((newSize - (inputs.shape[2]))/2)
        kernel_padding = int((newSize - (self.kernel_size[0]))/2)

        new_dim_im = [[0,0], 
                    [final_padding, final_padding-1],
                    [final_padding, final_padding-1], 
                    [0,0],
                    ]

        new_dim_ker = [[kernel_padding, kernel_padding],
                       [kernel_padding, kernel_padding], 
                       [0,0],
                       [0,0],
                     ]
        
        images = tf.pad(inputs, new_dim_im)
        kernel = tf.pad(self.kernel, new_dim_ker)
        
        images = tf.signal.rfft2d(images, [newSize, newSize])
        kernel = tf.signal.rfft2d(kernel, [newSize, newSize])
        kernel = tf.transpose(kernel, [2, 0, 1, 3])
        print(f'{images.shape=}')
        print(f'{kernel.shape=}')
        fft_image = images*kernel
        output = tf.signal.irfft2d(fft_image, fft_length=[newSize-1, newSize-1])
        return output

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
