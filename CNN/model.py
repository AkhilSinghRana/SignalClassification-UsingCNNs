######################
#   CNN model Architecture
######################
# We use tf 2.0 and tf.Keras API

import tensorflow as tf

from tf.keras.models import Sequential
import tf.keras.layers as layers # import Keras Layers
import tf.keras.optimizers as optimizers

#Load data generator and preprocessing module
from tf.keras.preprocessing.image import ImageDataGenerator

#import os module
import os

#import other modules
import numpy as np

class CNN_model():
    
    # Initialize the class, with some default variables
    def __init__(self, args=None):
        """
            args = Argument parser from the options.py file
        """
        self.args = args

    # This function defines the CNN architecture
    def cnn_model(self):
        model = Sequential()
        model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = 1, padding='same',
                 input_shape=(self.args.b, self.args.img_h, self.args.img_w, self.args.num_channels)))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(128, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
        print("Model Details!")
        model.summary()
        return model
