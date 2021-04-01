import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images_list = load_images_from_folder('images')

# plagiarized (borrowed) from
# "Honestly stolen"
# https://www.codegrepper.com/code-examples/python/how+to+read+all+images+from+a+folder+in+python+using+opencv

'''
# setup
IMG_SIZE = (10, 10) # arbitrary value; 
xsize = 10
ysize = 10
DATA_DIR = r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\images'

# get a list of files to load
f = []
for (dirpath, dirnames, filenames) in walk(DATA_DIR):
    f.extend(filenames)
    break

# TO DO: test whether the os.walk works
print(f)


# load the images and make them grayscale
for filename in f:
    actual_filename = path.join("images", filename)
    print(actual_filename)
    image = cv2.imread(actual_filename)
    print(image.shape())
    rescaled_image = cv2.resize(image, (xsize, ysize))
    # rescaled_image = cv2.resize(image, (xsize, ysize), interpolation = cv2.INTER_AREA)
    #rescaled_image = cv2.cvtColor(scr=rescaled_image, code=cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('dataset/' + 'make_filename_here' + '.jpg', rescaled_image)
    
    

import cv2
import numpy as np

img = cv2.imread('messi5.jpg')

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='/path/to/folder'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

import glob
import cv2

images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]

'''