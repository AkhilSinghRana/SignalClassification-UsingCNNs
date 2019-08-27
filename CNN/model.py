######################
#   CNN model Architecture
######################
# We use tf 2.0 and tf.Keras API

import tensorflow as tf

from tf.keras.models import Sequential
import tf.keras.layers as layers # import Keras Layers

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
        model.add(layers.Conv2d(filters))

        return model
