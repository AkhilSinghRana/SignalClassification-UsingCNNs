import argparse

def parseArguments():
	parser = argparse.ArgumentParser()# Creates argument parser

	
	parser.add_argument('--mode', help='Training or Testing/Inference modes', default="train", type=str)
	
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
    parser.add_argument('--save_dir', help='Path to save/load the trianed Models', default='./models')

	# Training Parameters
	parser.add_argument('--img_h', help='Image Height', type=int, default=512)
    parser.add_argument('--img_w', help='Image width', type=int, default=512)
	parser.add_argument('-b', '--batch_size', help='Batch Size for training', type=int, default=1)
	parser.add_argument('--num_epochs', help='Number of Epochs to train the model for', type=int, default= 100)
	parser.add_argument('--early_stop', help='EarlyStopping, to avoid Overfitting duting training ', action='store_true')




	args = parser.parse_args()
	return args
