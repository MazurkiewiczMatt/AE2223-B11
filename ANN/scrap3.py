import numpy
import cv2
import os
import pandas

# process:
# 1. import images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images

images = load_images('C:/Users/ic120/Desktop/AE2223-B11/ANN/images_fleet')
# produces list of arrays where each array is an image

# 2. resize images and make them grayscale
xsize = 100
ysize = 100
resized_images = []
grayscale_resized_images = []

for i in images:
    resized_images.append(cv2.resize(i, (xsize,ysize)))

for z in resized_images:
    grayscale_resized_images.append(cv2.cvtColor(z, cv2.COLOR_BGR2GRAY))

# 3. add type label; the images in folder were grouped, thus the manual label addition could be done as follows:
label_numbers = numpy.array([0]*10 + [1]*12 + [2]*10 + [3]*9 + [4]*18 + [5]*8 + [6]*10 + [7]*10 + [8]*3 + [9]*10 )

# 4. create dataset for further analysis as an array
dataset = numpy.vstack(grayscale_resized_images)
dataset = dataset.T # transposed so that there's 100 rows, 1 row per image, and each pixel is a column

y = numpy.insert(dataset, 0, label_numbers, axis=1) # add labels to dataset array

# 5. export data of the array for the ANN to study from
df = pandas.DataFrame(y)
df.to_csv('y.csv')