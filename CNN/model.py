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

