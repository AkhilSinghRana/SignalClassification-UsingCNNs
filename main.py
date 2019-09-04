from CNN import generate_model
from util import dataLoader
from tensorflow.keras import models as keras_model
from tensorflow import keras as keras
import tensorflow_hub as hub

import numpy as np
import matplotlib.pylab as plt

import options

def train(args):
    
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    # Create CNN model to train
    print("Generating CNN Model ....")
    model_obj = generate_model.Generate_model(args, dataloader.num_classes)
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
    if not args.usePreTrain:
        reloaded = keras.experimental.load_from_saved_model(args.load_dir, custom_objects={'KerasLayer':hub.KerasLayer})
        reloaded.summary()
        
        class_names = sorted(test_data_gen.class_indices.items(), key=lambda pair:pair[1])
        class_names = np.array([key.title() for key, value in class_names])
        
        print(class_names)
        
        #Do the prediction of batches
        for image_batch, label_batch in test_data_gen:
            predicted_batch = reloaded.predict(image_batch)
            predicted_id = np.argmax(predicted_batch, axis=-1)
            predicted_label_batch = class_names[predicted_id]

            print("Predicted -->", predicted_label_batch)
            # Actual label id
            label_id = np.argmax(label_batch, axis=-1)
            print("Real GT Batch-->", label_id)

            plt.figure(figsize=(10,9))
            plt.subplots_adjust(hspace=0.5)
            for n in range(30):
                plt.subplot(6,5,n+1)
                plt.imshow(image_batch[n])
                color = "green" if predicted_id[n] == label_id[n] else "red"
                plt.title(predicted_label_batch[n].title(), color=color)
                plt.axis('off')
                _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
            plt.show()
            


#use Pre Trained MOdels for Fine Tuning
def usePreTrain(args):
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    # Generate the model
    generateModel = generate_model.Generate_model(args, dataloader.num_classes)
    model = generateModel.preTrainedModel()
    history = model.fit(train_data_gen, epochs=2,
                steps_per_epoch= dataloader.num_train_images // args.batch_size)
    
    keras.experimental.export_saved_model(model, args.save_dir)

    

if __name__ == "__main__":
    args = options.parseArguments()
    
    if args.mode == "usePreTrain":
        usePreTrain(args)
    
    
    elif args.mode=="train":
        train(args)

    elif args.mode=="continueTrain":
        continueTrain(args)

    elif args.mode=="test":
        test(args)

    else:
        raise NotImplementedError