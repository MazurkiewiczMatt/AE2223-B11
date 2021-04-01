import cv2 # TO DO: pip install opencv-python
import os


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

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


xsize = 100
ysize = 100

DATA_DIR = r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\images'

# get a list of files to load
f = []
for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
    f.extend(filenames)
    break
paths_list = []
for i in range(100):
    paths_list.append(get_folder_file('images', f[i]))

for filename in paths_list:
    #actual_filename = path.join("images", filename)
    #print(actual_filename)
    image = cv2.imread(filename)
    rescaled_image = cv2.resize(image, (xsize, ysize))
    # rescaled_image = cv2.resize(image, (xsize, ysize), interpolation = cv2.INTER_AREA)
    rescaled_image = cv2.cvtColor(src=rescaled_image, code=cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\ANN\image.png', rescaled_image)

'''
images_list = load_images_from_folder('images/')

# https://stackoverflow.com/questions/38675389/python-opencv-how-to-load-all-images-from-folder-in-alphabetical-order
# it works for them???????? magic 
print(images_list)'''