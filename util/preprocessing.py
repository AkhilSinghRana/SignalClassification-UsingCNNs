import glob, os, sys
sys.path.append(".")
sys.path.append("..")
import options

def splitTrainValTest(dataset_dir):
    classses_to_include = [""]
    dir = [ dir for dir in os.listdir(dataset_dir) if dir!="dataset"]
    print(dir)
    
    
    
if __name__ == "__main__":
    args = options.parseArguments()
    splitTrainValTest(args.input_dir)