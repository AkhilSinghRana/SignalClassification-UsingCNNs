######################
#   CNN model Architecture
######################
# We use tf 2.0 and tf.Keras API

import tensorflow as tf

from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers # import Keras Layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras as keras

# Import Tensorflow_hub for using pre trained models
import tensorflow_hub as hub

#Load data generator and preprocessing module
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import os module
import os

#import other modules
import numpy as np

class Generate_model():
    
    # Initialize the class, with some default variables
    def __init__(self, args=None, num_classes=0):
        """
            args = Argument parser from the options.py file
        """
        self.args = args
        self.num_classes = num_classes

    # This function defines the CNN architecture
    def cnn_model(self):
        model = Sequential()
        model.add(layers.Conv2D(filters = 32, kernel_size = (5, 5), strides = 1, padding='same',
                 input_shape=(self.args.img_h, self.args.img_w, self.args.num_channels)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.LayerNormalization())
        model.add(layers.Dropout(0.25))
        
        # model.add(layers.Conv2D(64, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(64, (3, 3)))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        # model.add(layers.Dropout(0.5))
        
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        
        # model.add(layers.Conv2D(128, (3, 3)))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        # model.add(layers.Conv2D(256, (3, 3)))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        # model.add(layers.Dropout(0.5))
        
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())
        # model.add(layers.Conv2D(256, (3, 3), padding='same'))
        # model.add(layers.Activation('relu'))
        # model.add(layers.LayerNormalization())        
        


        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Model compile defines the otimizer and loss function to choose
        if self.args.optimizer=="RMS":
            model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_hinge",metrics=["accuracy"])
        elif self.args.optimizer=="Adam":
            model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])
        else:
            raise NotImplementedError
        
        # Generate the model Summary
        print("Model Details!")
        model.summary()
        return model

    def preTrainedModel(self):
        #model_name = (args.pre_trained_model_name, args.img_h) #@param ["(\"mobilenet_v2\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
        model_url = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(self.args.pre_trained_model_name)


        do_fine_tuning = not self.args.freeze_feature_layers
        print("Do Fine Tuning of feature layers : ",do_fine_tuning)
        

        feature_extractor_layer = hub.KerasLayer(model_url,
                                            input_shape=(self.args.img_h, self.args.img_w,3), trainable=do_fine_tuning)
        print("Building model from --> ", model_url)
        model = keras.Sequential([

                        feature_extractor_layer,
                        keras.layers.Dropout(rate=0.2),
                        keras.layers.Dense(self.num_classes, activation='softmax',
                                            kernel_regularizer=keras.regularizers.l2(0.0001))
                    ])

        # Generate Model Summary
        model.summary()
        
        #Compile the model for training
        print("Using {} Optimizer for training".format(self.args.optimizer))
        if self.args.optimizer=="RMS":
            model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),metrics=["accuracy"])
        elif self.args.optimizer=="Adam":
            model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        elif self.args.optimizer=="SGD":
            model.compile(optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9), 
                            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
        else:
            raise NotImplementedError
        
        return model