from CNN import model as model
from util import dataLoader
from tensorflow.keras import models as keras_model

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

    # Load and continue training from the saved model
    print("Loading model")
    loaded_model = keras_model.load_model(args.load_dir)

    #Predict from the model
    predictions = loaded_model.predict_generator(test_data_gen, steps = test_data_gen.n // args.batch_size, verbose=1)
    
    predicted_class_indices = np.argmax(predictions, axis=1)

    print(predictions, predicted_class_indices)

if __name__ == "__main__":
    args = options.parseArguments()
    
    if args.mode=="train":
        train(args)

    elif args.mode=="continueTrain":
        continueTrain(args)

    elif args.mode=="test":
        test(args)

    else:
        raise NotImplementedError