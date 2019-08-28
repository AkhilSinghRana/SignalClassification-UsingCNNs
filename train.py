from CNN import model as model
from util import dataLoader

import options

def train(args):
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    data_generator = dataloader.dataGenerator()
    
    print("Generating CNN Model ....")
    model_obj = model.CNN_model(args)
    cnn_model = model_obj.cnn_model()
    
    
if __name__ == "__main__":
    args = options.parseArguments()
    train(args)
    