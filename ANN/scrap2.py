import numpy
import cv2
import os
import pandas

# process:
# 1. import images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\images')
# produces list of arrays

# 2. resize images and make them grayscale
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
names = os.listdir(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\images')
for n in names:
    label_list.append(n.split('('))
flat_label_list = list(numpy.concatenate(label_list).flat)
flat_label_list = flat_label_list[::2]

label_numbers = numpy.array([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10+ [7]*10 + [8]*10 + [9]*10)

# 4 create dataset for further analysis as an array
dataset = numpy.vstack(grayscale_resized_images)
dataset = dataset.T # so that there's 100 rows and each pixel is a column

x = numpy.insert(dataset, 0, label_numbers, axis=1)
df = pandas.DataFrame(x)
df.to_csv('x.csv')
