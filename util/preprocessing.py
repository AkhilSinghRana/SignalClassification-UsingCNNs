import glob, os, sys, random
from subprocess import call
import numpy as np
from skimage import io, transform, util

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
                if num_moved_images < 1300:
                    print("Moving Images to Train in progress -->", num_moved_images)
                    call(['cp',img, dataset_dir+"/dataset/train/"+dir])
                    
                    #Uncomment to create validations set
                    """elif num_moved_images < 2500+200:
                        print("Moving Images to Validation in progress -->", num_moved_images)
                        call(['cp',img, dataset_dir+"/dataset/val/"+dir])
                    """    
                else:
                    print("Moving Images to Test in progress -->", num_moved_images)
                    call(['cp',img, dataset_dir+"/dataset/test/"+dir])
                
                num_moved_images+=1
                    
# This function creates a dummy waterfall representation from differenet signals
#           It selects the signals randomly, rotates it by 90 degrees and stitch them next to each other

def createWaterfall_likeRepresentation(input_dir):
    dirs = [os.path.join(input_dir,dir) for dir in os.listdir(input_dir) if not dir.startswith("_") and os.path.isdir(os.path.join(input_dir,dir))]
    print("Found {} classes in the input dir".format(len(dirs)))
    
    images = []
    for dir in dirs:
        images += glob.glob(dir+"/*.jpg")

    #Shuffle the list to select random classes from the list        
    random.shuffle(images)
    random.shuffle(images)
    
    
    for i in range(0, 8, 4):#len(images)-4):
        selected_images = images[i:i+4]
        print(len(selected_images))
        for j, image in enumerate(selected_images):
            # read, resize and then rotate the array 90 degrees counter clockwise!
            image_array =  np.rot90(transform.resize( util.crop(io.imread(image, as_gray= True), ((50,75),(113,82)) ), output_shape = (256,256), preserve_range=True))
            
            waterfall_image_array = image_array if j == 0 else np.hstack((waterfall_image_array, image_array))

        io.imsave("waterfallRepresentation_{}.png".format(i),waterfall_image_array)
        break
    
    
if __name__ == "__main__":
    args = options.parseArguments()
    #splitTrainValTest(args.input_dir)

    createWaterfall_likeRepresentation(args.input_dir)