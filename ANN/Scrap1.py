import cv2
import os
import numpy as np

x = np.array([[1, 3, 5, 3],
             [5, 4, 3, 3]])
y = np.where(x == np.array([1, 3, 5, 3]))
print(y)
