import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import rosbag


def bagselect(bagnum):
    # input trajectory coordinates, in RHS
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    with rosbag.Bag("Bags/" + bagnum) as bag:
        for topic, msg, t in bag.read_messages(topics=['/optitrack/pose']):
            x_coordinates.append(msg.pose.position.x)
            y_coordinates.append(msg.pose.position.y)
            z_coordinates.append(msg.pose.position.z)
    return x_coordinates, y_coordinates, z_coordinates


obstacle_data = pd.read_csv('C:/Users/ic120/PycharmProjects/AE2223-B11/Bags/trial_overview.csv')
# input column coordinates, in RHS
x_column = obstacle_data["Obstacle x"]
x_column = x_column.drop_duplicates()
y_column = obstacle_data["Obstacle y"]
y_column = y_column.drop_duplicates()
z_column = obstacle_data["Obstacle z"]
z_column = z_column.drop_duplicates()

# column coordinates in single array, x-y-z for the 3 variants
column_variants = np.stack((x_column, y_column, z_column), axis=1)
Xc1, Zc1, Yc1 = column_variants[0]  # first column
Xc2, Zc2, Yc2 = column_variants[1]  # second column
Xc3, Zc3, Yc3 = column_variants[2]  # third column

# making a 2D plot:
figure, axes = plt.subplots()
radius = 0.4
c = plt.Circle((Xc1, Yc1), radius, color="orange", alpha=0.5)
cc = plt.Circle((Xc2, Yc2), radius, color="r", alpha=0.5)
ccc = plt.Circle((Xc3, Yc3), radius, color="b", alpha=0.5)

x, y, z = bagselect("41.bag")

plt.plot(x, z, color= "g")
plt.xlim((5, -5))
plt.ylim((-5, 5))
plt.xlabel("x-coordinate position [m]", fontsize = 15, fontname = "DejaVu Sans")
plt.ylabel("z-coordinate position [m]", fontsize = 15, fontname = "DejaVu Sans")
axes.add_artist(c)
axes.add_artist(cc)
axes.add_artist(ccc)
plt.show()
