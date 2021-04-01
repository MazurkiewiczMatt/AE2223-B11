import cv2 # TO DO: pip install cv2
from os import walk

# setup
IMG_SIZE = (1000, 500) # arbitrary value

# get a list of files to load
f = []
for (dirpath, dirnames, filenames) in walk('dataset_raw'):
    f.extend(filenames)
    break

# TO DO: create the 'dataset_raw' folder with the images
# TO DO: test whether this works
print(f)


# load the images
for filename in f:
    image = cv2.imread(filename, 0)
    rescaled_image = cv2.resize(image, IMG_SIZE, interpolation = cv2.INTER_CUBIC)
    # TO DO: save the image to 'dataset' folder
    
