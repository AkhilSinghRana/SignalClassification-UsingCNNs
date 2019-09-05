import os, glob
from tensorflow.keras.preprocessing import image as image_preprocessing

class DataLoader():
    def __init__(self, args=None):
        print("Initializing DataLoader ...")
        self.args = args
        self.num_classes = 0 # Number of class to train on
        self.num_train_images = 0 # Number of training samples
        self.num_val_images = 0 # Number of validation samples
        self.num_test_images = 0 # number of test samples
        
    def dataGenerator(self):
        print("Looking for the data at -->", self.args.input_dir)
        if not os.path.exists(self.args.input_dir):
            raise NotADirectoryError
        
        else:
            if self.args.mode == "train" or self.args.mode == "continueTrain" or self.args.mode == "usePreTrain":
                train_dir = os.path.join(self.args.input_dir, "train")
                val_dir = os.path.join(self.args.input_dir, "val")
                if not (os.path.exists(train_dir) or os.path.exists(val_dir)):
                    print("Make sure that, train and Val folders are present!")
                    raise FileNotFoundError
                else:
                    print("Classes : {} -----------".format(os.listdir(train_dir)))
                    assert os.listdir(train_dir) == os.listdir(val_dir), "Training and Validation Classes have to be the same...."
                    
                    # Count number of training and Validation Images in the dataset
                    for dir in os.listdir(train_dir):
                        self.num_train_images += len(glob.glob(train_dir+"/{}/*.jpg".format(dir), recursive=True))
                        self.num_classes+=1
                    
                    for dir in os.listdir(val_dir):
                        self.num_val_images += len(glob.glob(val_dir+"/{}/*.jpg".format(dir), recursive=True))
                    
                    
                    # Create a data generator that generates the data on the fly with data augmentation
                    train_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255, horizontal_flip= True)
                    val_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255, horizontal_flip= True)
                    
                    # Data Generator with specific batch size from directory
                    print("Training Data Loader")
                    train_data_gen = train_image_gen.flow_from_directory(directory=train_dir,
                                                                         target_size = (self.args.img_h,self.args.img_w),
                                                                         color_mode = "rgb",
                                                                         batch_size = self.args.batch_size,
                                                                         shuffle=True,
                                                                         interpolation = "bicubic",
                                                                         class_mode = "categorical"
                                                    )
                    print("Validation Data Loader")
                    val_data_gen = val_image_gen.flow_from_directory(directory=val_dir,
                                                                         target_size = (self.args.img_h,self.args.img_w),
                                                                         color_mode = "rgb",
                                                                         batch_size = self.args.batch_size,
                                                                         shuffle=True,
                                                                         interpolation = "bicubic",
                                                                         class_mode = "categorical"
                                                    )
                    
                    return train_data_gen, val_data_gen
            
            elif self.args.mode == "test":
                test_dir = os.path.join(self.args.input_dir, "test")
                print("Evaluation mode")
                
                test_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255, horizontal_flip= True)

                #Create test data gen
                test_data_gen = test_image_gen.flow_from_directory(directory=test_dir,
                                                                         target_size = (self.args.img_h,self.args.img_w),
                                                                         color_mode = "rgb",
                                                                         batch_size = self.args.batch_size,
                                                                         shuffle=True,
                                                                         interpolation = "bicubic",
                                                                         class_mode = "categorical"
                                                                    )

                return test_data_gen

            else:
                raise NotImplementedError