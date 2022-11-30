import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
filedir = os.path.dirname(__file__)

tf.keras.utils.set_random_seed(1336)

TRAINDIR = filedir + "/../data/chest_xray/train"
TESTDIR = filedir + "/../data/chest_xray/test"
BATCHSIZE = 256
IMG_HEIGHT = 227
IMG_WIDTH = 227


def preprocess(img, label):
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) / 255, label

HEIGHT = 200
WIDTH = 200
split = ['train[:70%]', 'train[70%:]']

trainDataset, testDataset = tfds.load(name='cats_vs_dogs', split=split, as_supervised=True)

trainDataset = trainDataset.map(preprocess).batch(BATCHSIZE)
testDataset = testDataset.map(preprocess).batch(BATCHSIZE)

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
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(
            4096, activation="relu"),# kernel_regularizer=kernel, bias_regularizer=bias
        #),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(
            4096, activation="relu"), #, kernel_regularizer=kernel, bias_regularizer=bias
        #),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    trainDataset,
    #batch_size=BATCHSIZE,
    validation_data=testDataset,
    epochs=20,
    # class_weight=class_weight,
)
