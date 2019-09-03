from CNN import model as model
from util import dataLoader
from tensorflow.keras import models as keras_model
from tensorflow import keras as keras
import tensorflow_hub as hub

import numpy as np
import options

def train(args):
    
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    # Create CNN model to train
    print("Generating CNN Model ....")
    model_obj = model.CNN_model(args, dataloader.num_classes)
    cnn_model = model_obj.cnn_model()
    
    # Fit the dataset for training
    history = cnn_model.fit_generator(
                    train_data_gen,
                    steps_per_epoch=dataloader.num_train_images // args.batch_size,
                    epochs=args.num_epochs,
                    validation_data=val_data_gen,
                    validation_steps=dataloader.num_val_images // args.batch_size
                )
    keras_model.save_model(cnn_model, filepath=args.save_dir, overwrite=True, include_optimizer=True, save_format=args.model_save_format)

def continueTrain(args):
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    # Load and continue training from the saved model
    print("Loading model")
    loaded_model = keras_model.load_model(args.load_dir)

    # COntinue train and save again
    history = loaded_model.fit_generator(
                    train_data_gen,
                    steps_per_epoch=dataloader.num_train_images // args.batch_size,
                    epochs=args.num_epochs,
                    validation_data=val_data_gen,
                    validation_steps=dataloader.num_val_images // args.batch_size
                )
    keras_model.save_model(loaded_model, filepath=args.save_dir, overwrite=True, include_optimizer=True, save_format=args.model_save_format)

#Load the model for Inference
def test(args):
    print("Load for inference")
    
    dataloader = dataLoader.DataLoader(args)
    test_data_gen = dataloader.dataGenerator()

    #if use PreTrain flag is enabled
    if args.usePreTrain:
        reloaded = tf.keras.experimental.load_from_saved_model(args.load_dir, custom_objects={'KerasLayer':hub.KerasLayer})


#use Pre Trained MOdels for Fine Tuning
def usePreTrain(args):
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    #model_name = (args.pre_trained_model_name, args.img_h) #@param ["(\"mobilenet_v2\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
    model_url = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(args.pre_trained_model_name)


    do_fine_tuning = not args.freeze_feature_layers
    print("Do Fine Tuning of feature layers : ",do_fine_tuning)
    

    feature_extractor_layer = hub.KerasLayer(model_url,
                                         input_shape=(args.img_h, args.img_w,3), trainable=do_fine_tuning)
    print("Building model with", model_url)
    model = keras.Sequential([
                    feature_extractor_layer,
                    keras.layers.Dropout(rate=0.2),
                    keras.layers.Dense(train_data_gen.num_classes, activation='softmax',
                                        kernel_regularizer=keras.regularizers.l2(0.0001))
                ])
    model.summary()

    #Compile the model for training
    model.compile(optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9), 
                    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    
    history = model.fit(train_data_gen, epochs=2,
                steps_per_epoch= dataloader.num_train_images // args.batch_size)
    
    keras.experimental.export_saved_model(model, args.save_dir)

    


if __name__ == "__main__":
    args = options.parseArguments()
    if args.usePreTrain:
        usePreTrain(args)
    
    else:
        if args.mode=="train":
            train(args)

        elif args.mode=="continueTrain":
            continueTrain(args)

        elif args.mode=="test":
            test(args)

        else:
            raise NotImplementedError