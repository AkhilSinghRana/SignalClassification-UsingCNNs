import argparse

def parseArguments():
    # Creates argument parser
    parser = argparse.ArgumentParser()

    # Boolean to utilize  PreTrained Networks from tf-Hub
    parser.add_argument('--usePreTrain', help='Enable the flag if you want to use pre trained Networks!', action="store_true")
    parser.add_argument('pre_trained_model_name', help='Name of the pretrained model to be used, defaults to Inception v3', type=str, 
                                default="inception_v3")
    #Only if name is None                                
    parser.add_argument('pre_trained_model_url', help='URL of the pretrained model to be used, defaults to Inception v3, this can also replaced dynamically with the name above', type=str, 
                                default="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format("inception_v3"))


    # Load the model for training testing or to continue the training from a specific checkpoint!
    parser.add_argument('--mode', help='train/continueTrain/test modes', default="train", type=str)


    # Path to the dataset Directory. It should have the below Folder Structure

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

    parser.add_argument('--input_dir', help='Path to the Dataset directory(DIR)', required=True)

    # Tensorflow Graph,Session,model based parametes
    parser.add_argument('--save_dir', help='Path to save the trained Models', default='./models')
    parser.add_argument('--load_dir', help='Path to load the trained Models', default='./models')
    parser.add_argument('--continue_train', help='Resumes the training from the latest checkpoint, mentioned in load_dir', action="store_true")
    parser.add_argument('--model_save_format', help='save the model in tf or h5py format', default="tf", type=str)


    # Training Parameters
    parser.add_argument('--img_h', help='Image Height', type=int, default=512)
    parser.add_argument('--img_w', help='Image width', type=int, default=512)
    parser.add_argument('--num_channels', help='Image Channels (1 or 3) defaults to 3', type=int, default=3)
    parser.add_argument('-b', '--batch_size', help='Batch Size for training', type=int, default=1)
    parser.add_argument('--num_epochs', help='Number of Epochs to train the model for', type=int, default= 100)
    parser.add_argument('--early_stop', help='EarlyStopping, to avoid Overfitting duting training ', action='store_true')
    parser.add_argument('--optimizer', help="Optimizer to Choose RMS or Adam", default="RMS", type=str)

    args = parser.parse_args()
    return args