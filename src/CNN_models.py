
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from functools import partial

"""
This is a script similiar to utils.py containing the architectures mentioned by Geron in his book 
'Hands On Machine Learning with SciKit-learn and Tensorflow" which we use throughout this project.
"""


# --------------------------------------------- Alex-net ------------------------------------------------------
kernel = tf.keras.regularizers.L2(l2=0.001)
bias = tf.keras.regularizers.L2(l2=0.001)

model = tf.keras.Sequential(
    [
        #tf.keras.layers.Rescaling(1.0 / 255),
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
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            4096, activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias
        #),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            4096, activation="relu"), #, kernel_regularizer=kernel, bias_regularizer=bias
        #),
        tf.keras.layers.Dense(1),
    ]
)

# ------------------------------------------ ResNet-34 CNN ---------------------------------------------------

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding='same', kernel_initializer='he_normal', 
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
                DefaultConv2D(filters, strides=strides), 
                tf.keras.layers.BatchNormalization(),
                self.activation,
                DefaultConv2D(filters),
                tf.keras.layers.BatchNormalization()
                ]
        self.skip_layers = []
        if strides > 1: 
            self.skip_layers = [
                    DefaultConv2D(filters, kernel_size=1, strides=strides),
                    tf.keras.layers.BatchNormalization()
                ]
    
    def call(self, inputs): 
        Z = inputs
        for layer in self.main_layers: 
            Z = layer(Z) 
        skip_Z = inputs
        for layer in self.skip_layers: 
            skip_Z = layer(skip_Z) 
        return self.activation(Z + skip_Z)




