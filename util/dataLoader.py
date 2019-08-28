import os
class DataLoader():
    def __init__(self, args=None):
        print("Initializing DataLoader ...")
        self.args = args
        
    def dataGenerator(self):
        print("Looking for the data at -->", self.args.input_dir)
        if not os.path.exists(self.args.input_dir):
            raise NotADirectoryError
        
        else:
            if self.args.mode == "train":
                print("Directory found")
                train_dir = os.path.join(self.args.input_dir, "train")
                val_dir = os.path.join(self.args.input_dir, "val")
                if not (os.path.exists(train_dir) or os.path.exists(val_dir)):
                    print("Make sure that, train and Val folders are present!")
                    raise FileNotFoundError
                else:
                    print("----------  Number Of Classes : {} -----------".format(os.listdir(train_dir)))
                    assert os.listdir(train_dir) == os.listdir(val_dir), "Training and Validation Classes have to be the same...."
                    
                
            else:
                test_dir = os.path.join(self.args.input_dir, "test")