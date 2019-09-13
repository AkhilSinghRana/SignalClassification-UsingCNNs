import glob, os, sys, random
from subprocess import call
import numpy as np
from skimage import io, transform, util, img_as_uint
from skimage.color import gray2rgb

#Imports for creating XML documents!
import xml.etree.ElementTree as ET
from xml.dom import minidom # Used later for pretty printing with proper spaces between root and child

# Load Parse the Options from terminal!
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
                    
                    #Uncomment to create validations set
                    """elif num_moved_images < 2500+200:
                        print("Moving Images to Validation in progress -->", num_moved_images)
                        call(['cp',img, dataset_dir+"/dataset/val/"+dir])
                    """    
                else:
                    if num_moved_images<4000:
                        print("Moving Images to Test in progress -->", num_moved_images)
                        call(['cp',img, dataset_dir+"/dataset/test/"+dir])
                    else:
                        break
                num_moved_images+=1
                    
# This function creates a dummy waterfall representation from differenet signals
#           It selects the signals randomly, rotates it by 90 degrees and stitch them next to each other

def createWaterfall_likeRepresentation(args, input_dir):
    dirs = [os.path.join(input_dir,dir) for dir in os.listdir(input_dir) if not dir.startswith("_") and os.path.isdir(os.path.join(input_dir,dir))]
    num_classes = len(dirs)
    print("Found {} classes in the input dir".format(num_classes))
    
    images = []
    for dir in dirs:
        images += glob.glob(dir+"/*.jpg")

    #Shuffle the list to select random classes from the list        
    random.shuffle(images)
    random.shuffle(images)
    
    num_rows = 3 # Number of rows to create, for waterfall equivalent to time scale!
    # randomly generate a row number for each class, the signal will then be placed at that row while creating waterfall
    row_numbers = [random.randint(0,num_rows-1) for i in range(num_classes) ]

    # Get the noise from cropped Unknown signals, to match the actual signal noise!
    noise_1 = transform.resize( io.imread("noise{}.jpg".format(row_numbers[0]+1), plugin="imageio", as_gray=True), output_shape = (args.img_h,args.img_w ))
    noise_2 = transform.resize( io.imread("noise{}.jpg".format(row_numbers[1]+1), plugin="imageio", as_gray=True), output_shape = (args.img_h,args.img_w ))
    noise_3 = transform.resize( io.imread("noise{}.jpg".format(row_numbers[2]+1), plugin="imageio", as_gray=True), output_shape = (args.img_h,args.img_w ))
    noise_4 = transform.resize( io.imread("noise{}.jpg".format(row_numbers[3]+1), plugin="imageio", as_gray=True), output_shape = (args.img_h,args.img_w ))
    noise = np.empty(shape= (args.img_h * num_rows, args.img_w * num_classes))
    for i in range(num_rows):
        start_row_pixel = i * args.img_h
        noise[start_row_pixel: start_row_pixel + args.img_h  , :] = np.hstack( (noise_1, noise_2, noise_3, noise_4) ) 
    
    # Save the noisy image for testing
    #io.imsave("noise.jpg" ,noise)

    images_output_folder = os.path.join(args.output_dir, "images")
    annotations_output_folder = os.path.join(args.output_dir, "annotations")
    xmls_output_folder = os.path.join(annotations_output_folder, "xmls")
    if not os.path.isdir(images_output_folder):
        os.makedirs(images_output_folder)
        os.makedirs(annotations_output_folder)
        os.makedirs(xmls_output_folder)

    train_val_filenames = []
    for i in range(0, len(images)-num_classes, num_classes):#:
        selected_images = images[i:i+num_classes]
        print(row_numbers)
        WR_signal = np.copy(noise)
        bnd_box = []
        for j, image in enumerate(selected_images):
            # read --> crop --> -->  resize --> rotate the array 90 degrees counter clockwise!
            image_array =  np.rot90(transform.resize( util.crop(io.imread(image, plugin="imageio", as_gray=True), ((51,76),(113,82)) ), output_shape = (args.img_h,args.img_w ), preserve_range=True))
            

            # Replace the part of noisy image with Signal image 
            row_number = num_classes-2 #row_numbers[j] For randomization
            start_row_pixel = row_number * args.img_h
            start_col_pixel = j*args.img_w
            
            if random.random() < 0.5: # Save waterfall image with probability
                WR_signal[start_row_pixel: start_row_pixel + args.img_h , start_col_pixel : start_col_pixel + args.img_w] = image_array
            
            #waterfall_image_array = image_array if j == 0 else np.hstack((waterfall_image_array, image_array))
            
            # append xmin, ymin and assume size of bounding box to be args.img_h
            class_name = image.split("/")[-2]
            print(class_name)
            bnd_box.append((class_name, start_col_pixel, start_row_pixel))

        #waterfall_image_array = None
        #noise = img_as_uint(noise)
        file_name = "WR_img_{}".format(int(i/num_classes+1))
        train_val_filenames.append(file_name)
        WR_signal = gray2rgb(WR_signal)
        #io.imsave("{}/{}.jpg".format(images_output_folder, file_name ), waterfall_image_array, plugin = "imageio")
        io.imsave("{}/{}.jpg".format(images_output_folder, file_name ), WR_signal, plugin = "imageio")
        #print(noise.shape)
        row_numbers = [random.randint(0,num_rows-1) for i in range(num_classes) ]
        # Code snippet below stores required variables and generate XML
        createXML(args=args, file_name=file_name, size=WR_signal.shape, bnd_box=bnd_box)
    
    io.imsave("noise.jpg", noise)
    with open(args.output_dir+"annotations/trainval.txt", "w") as f:
        f.writelines("\n".join(train_val_filenames))
    
def createXML(args, file_name=None, **kwargs):
    # The XML is created , for ObjectDetectionAPI, similar to labelIMG generated XML, refer Github repo!
    print("creating xml for -->", file_name)
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder" )
    folder.text = "annotation"
    
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = file_name+".jpg"
    
    path = ET.SubElement(annotation, "path")
    path.text = os.path.join(args.output_dir, "images/"+file_name+".jpg")

    if kwargs.keys is not None:
        # IMAGE Size related TAGS
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(kwargs.get("size")[1])
        height = ET.SubElement(size, "height")
        height.text = str(kwargs.get("size")[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(kwargs.get("size")[2])
    
        # OBJECT BOunding  Box Tags
        num_classes = 4
        for i in range(num_classes):
            object_i = ET.SubElement(annotation, "object")
            bnd_box_values = kwargs.get("bnd_box")[i]
            name = ET.SubElement(object_i, "name")
            name.text = bnd_box_values[0]
            bnd_box = ET.SubElement(object_i, "bndbox")
            
            xmin = ET.SubElement(bnd_box,"xmin")
            xmin.text = str(bnd_box_values[1])
            ymin = ET.SubElement(bnd_box,"ymin")
            ymin.text = str(bnd_box_values[2])
            xmax = ET.SubElement(bnd_box,"xmax")
            xmax.text = str(bnd_box_values[1] + args.img_w )# assuming bndBox size to be image_h
            ymax = ET.SubElement(bnd_box,"ymax")
            ymax.text = str( bnd_box_values[2] + args.img_h )# assuming bndBox size to be image_w

    

    # Pretify the XML
    root_string = ET.tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(root_string)
    Pretty_XML = reparsed.toprettyxml(indent="  ")

    #Wrtie the tree
    xml_filename = os.path.join(args.output_dir, "annotations/xmls/"+file_name + ".xml")
    with open(xml_filename, "w") as f:
        f.write(Pretty_XML)


if __name__ == "__main__":
    args = options.parseArguments()
    #splitTrainValTest(args.input_dir)

    createWaterfall_likeRepresentation(args, args.input_dir)