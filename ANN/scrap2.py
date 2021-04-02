import numpy
import cv2
import os
import pandas
# so uhh the other file broke my brain a bit so i decided to try to write something too
# with comments. so that it wouldn't break my brain as much

# process:
# 1 import images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder('C:/Users/ic120/Desktop/AE2223-B11/ANN/images')
# produces list of arrays

# 2 resize images and make them grayscale
xsize = 100
ysize = 100
resized_images = []
grayscale_resized_images = []

for i in images:
    resized_images.append(cv2.resize(i, (xsize,ysize)))

for z in resized_images:
    grayscale_resized_images.append(cv2.cvtColor(z, cv2.COLOR_BGR2GRAY))

# 3 add type label based on name
label_list = []
names = os.listdir('C:/Users/ic120/Desktop/AE2223-B11/ANN/images')
for n in names:
    label_list.append(n.split('('))
flat_label_list = list(numpy.concatenate(label_list).flat)
flat_label_list = flat_label_list[::2]

ynot = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,
        9,9,9,9,9,9,9,9,9,9,0,0,0,0,0,0,0,0,0,0] # i wanted to like, replace all types by digits at first
                                                # but then i realized it was gonna be longer than just. writing numbers like this
                                                # they're in alphabetic order anyway
ynotarray = numpy.array(ynot, dtype=int)

# 4 create dataset for further analysis as an array
dataset = numpy.vstack(grayscale_resized_images)
dataset = dataset.T # so that there's 100 rows and each pixel is a column

x = numpy.concatenate(ynotarray, dataset)
# my plan was then to join the arrays using np.concatenate((),axis=1) to make the huge array for data analysis
# that would sorta match his OG input with label first and pixels later
# BUT it looks like the dataset array is filled with str and not int ?? based on the TypeErrors
# TypeError: only integer scalar arrays can be converted to a scalar index