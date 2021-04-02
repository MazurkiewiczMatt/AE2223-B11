import cv2 
import os
import numpy as np


def get_file(file):
    # Returns the full path, if the file is in the same folder as the main .py program.
    # Useful if a computer uses some random directory (like mine)
    path = os.path.join(os.path.dirname(__file__), file)
    return path


def get_folder_file(folder, file):
    # Returns the full path, if the file is not in the same folder as the main .py program.
    # Useful if a computer uses some random directory (like mine)
    extension = os.path.join(folder, file)
    path = get_file(extension)
    return path


def loadimgs(save=False):
    # SETUP

    # Common image size:
    xsize = 100
    ysize = 100

    DATA_DIR = r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\images'

    images = np.array([])

    # get a list of files to load
    f = []
    for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
        f.extend(filenames)
        break

    images = np.array([0, 0])
    k = 0
    for i in f:
        filename = get_folder_file('images', i)
        image = cv2.imread(filename)
        rescaled_image = cv2.resize(image, (xsize, ysize))
        if save:
            cv2.imwrite(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\dataset_color\\' + i, rescaled_image)
        rescaled_image = cv2.cvtColor(src=rescaled_image, code=cv2.COLOR_BGR2GRAY)
        if save:
            cv2.imwrite(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\dataset\\' + i, rescaled_image)
        flat_img = rescaled_image.flatten()
        name_plane = i.split()[0] 
        if name_plane == 'A380':
            label = 0
        elif name_plane == 'AN225':
            label = 1
        elif name_plane == 'B747':
            label = 2
        elif name_plane == 'B737':
            label = 3
        elif name_plane == 'Beluga':
            label = 4
        elif name_plane == 'C130':
            label = 5
        elif name_plane == 'F16':
            label = 6
        elif name_plane == 'Fokker':
            label = 7
        elif name_plane == 'PHLAB':
            label = 8
        elif name_plane == 'SS100':
            label = 9

        images = np.vstack((images, np.array([flat_img, label])))
        if k == 0:
            np.delete(images, 0, 0)
        rescaled_image = cv2.Canny(rescaled_image,100,200)
        
        if save:
            cv2.imwrite(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\dataset_edge\\' + i, rescaled_image)
        k += 1
    print(images)
    print(np.shape(images))
    return images

loadimgs(False)
