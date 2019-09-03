import glob, os, sys, random
from subprocess import call
sys.path.append(".")
sys.path.append("..")
import options

def splitTrainValTest(dataset_dir):
    if not os.path.isdir(dataset_dir):
        os.makedirs(os.path.join(dataset_dir,'dataset/train'))
        os.makedirs(os.path.join(dataset_dir,'dataset/val'))
        os.makedirs(os.path.join(dataset_dir,'dataset/test'))
    
    
    for dir in os.listdir(dataset_dir):
        if dir!="dataset" and not dir.startswith("_"):
            os.makedirs(os.path.join(dataset_dir, "dataset/train/"+dir))
            os.makedirs(os.path.join(dataset_dir, "dataset/val/"+dir))
            os.makedirs(os.path.join(dataset_dir, "dataset/test/"+dir))
            
            #Check the number of images in the Directory
            images = glob.glob("{}/{}/{}".format(dataset_dir, dir,"*.jpg"))
            random.shuffle(images)
            random.shuffle(images)
            num_moved_images=0
            for img in images:
                if num_moved_images < 2500:
                    print("Moving Images to Train in progress -->", num_moved_images)
                    call(['cp',img, dataset_dir+"/dataset/train/"+dir])
                    
                elif num_moved_images < 2500+200:
                    print("Moving Images to Validation in progress -->", num_moved_images)
                    call(['cp',img, dataset_dir+"/dataset/val/"+dir])
                    
                else:
                    print("Moving Images to Test in progress -->", num_moved_images)
                    call(['cp',img, dataset_dir+"/dataset/test/"+dir])
                
                num_moved_images+=1
                    
            
if __name__ == "__main__":
    args = options.parseArguments()
    splitTrainValTest(args.input_dir)