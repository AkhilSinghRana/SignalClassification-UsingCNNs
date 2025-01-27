import argparse

def parseArguments():
    # Creates argument parser
    parser = argparse.ArgumentParser()


    # Load the model for training testing or to continue the training from a specific checkpoint!
    parser.add_argument('--mode', help='train/continueTrain/test modes', default="train", type=str)
    
    parser.add_argument('--pre_trained_model_name', help='Name of the pretrained model to be used, defaults to Inception v3', type=str, 
                                default="inception_v3")
    parser.add_argument('--model_url', help='url of the model to use for pre-training', type=str, 
                                default="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4")
    
    parser.add_argument('--freeze_feature_layers', help='Option to use the model as bottleneck or FeatureExtractor', action='store_true')
   
    # Path to the dataset Directory. It should have the below Folder Structure, if not have a look at preprocessing.py file provided in util folder

    ################# Folder Structure:###########
    #    DIR:
    #       |-- Train
    #           |-- Class A
    #           |-- Class B
    #           |-- Class C
    #           |-- Class D
    #           |-- Class E
    #       |-- Val
    #           |-- Class A
    #           |-- Class B
    #           |-- Class C
    #           |-- Class D
    #           |-- Class E
    #       |-- Test
    #           |-- Class A
    #           |-- Class B
    #           |-- Class C
    #           |-- Class D
    #           |-- Class E
    ###############################################

    parser.add_argument('--input_dir', help='Path to the Dataset directory(DIR)', default='./', type=str)
    parser.add_argument('--output_dir', help='Path to save the preprocesed Dataset directory(DIR)', default='./', type=str)


    # Tensorflow Graph,Session,model based parametes
    parser.add_argument('--save_dir', help='Path to save the trained Models', default='./models')
    parser.add_argument('--load_dir', help='Path to load the trained Models', default='./models')
    parser.add_argument('--continue_train', help='Resumes the training from the latest checkpoint, mentioned in load_dir', action="store_true")
    parser.add_argument('--model_save_format', help='save the model in tf or h5py format', default="tf", type=str)


    # Training Parameters
    parser.add_argument('--img_h', help='Image Height', type=int, default=256)
    parser.add_argument('--img_w', help='Image width', type=int, default=256)
    parser.add_argument('--num_channels', help='Image Channels (1 or 3) defaults to 3', type=int, default=3)
    parser.add_argument('-b', '--batch_size', help='Batch Size for training', type=int, default=1)
    parser.add_argument('--num_epochs', help='Number of Epochs to train the model for', type=int, default= 100)
    parser.add_argument('--early_stop', help='EarlyStopping, to avoid Overfitting duting training ', action='store_true')
    parser.add_argument('--optimizer', help="Optimizer to Choose RMS or Adam", default="SGD", type=str)

    args = parser.parse_args()
    return args