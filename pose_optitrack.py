import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_rotation_matrix(Q):
    # z = a + bi + cj + dk = w + xi + zj + yk in our case bc of OptiTrack wiki
    a = Q[0]  #
    b = Q[1]  #
    c = Q[3]  #
    d = Q[2]  #
    # First row of the rotation matrix
    r00 = a ** 2 + b ** 2 - c ** 2 - d ** 2
    r01 = 2 * b * c - 2 * a * d
    r02 = 2 * b * d + 2 * a * c

    # Second row of the rotation matrix
    r10 = 2 * b * c + 2 * a * d
    r11 = a ** 2 - b ** 2 + c ** 2 - d ** 2
    r12 = 2 * c * d - 2 * a * b

    # Third row of the rotation matrix
    r20 = 2 * b * d - 2 * a * c
    r21 = 2 * c * d + 2 * a * b
    r22 = a ** 2 - b ** 2 - c ** 2 + d ** 2

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix


def data_for_cylinder_along_z(center_x, center_y, radius_c, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius_c*np.cos(theta_grid) + center_x
    y_grid = radius_c*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


data = pd.read_csv('C:/Users/ic120/PycharmProjects/AE2223-B11/1/optitrack-pose.csv')
obstacle_data = pd.read_csv('C:/Users/ic120/PycharmProjects/AE2223-B11/trial_overview.csv')

# input trajectory coordinates, in RHS
x_coordinates = data['pose.position.x']
y_coordinates = data['pose.position.y']
z_coordinates = data['pose.position.z']

timestamps = data['Time']

# input rotations in quaternion system
x_orientation = data['pose.orientation.x']
y_orientation = data['pose.orientation.y']
z_orientation = data['pose.orientation.z']
w_orientation = data['pose.orientation.w']

for i in range(len(x_coordinates)):
    Q = [w_orientation[i], x_orientation[i], y_orientation[i], z_orientation[i]]
    E = quaternion_rotation_matrix(Q)  # rotation matrices from quaternions

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

# making 2D plot:
figure, axes = plt.subplots()
radius = 0.1
c = plt.Circle((Xc1, Yc1), radius)
cc = plt.Circle((Xc2, Yc2), radius)
ccc = plt.Circle((Xc3, Yc3), radius)
plt.xlim((-5, 5))
plt.ylim((-5, 5))
axes.add_artist(c)
axes.add_artist(cc)
axes.add_artist(ccc)
plt.show()

# making 3D plot:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
height = 2

Xc, Yc, Zc = data_for_cylinder_along_z(Xc1, Yc1, radius, height)
Xcc, Ycc, Zcc = data_for_cylinder_along_z(Xc2, Yc2, radius, height)
Xccc, Yccc, Zccc = data_for_cylinder_along_z(Xc3, Yc3, radius, height)

ax.plot_surface(Xc, Yc, Zc)
ax.plot_surface(Xcc, Ycc, Zcc)
ax.plot_surface(Xccc, Yccc, Zccc)

fig.suptitle('3D Trajectory', fontsize=20)

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(0, 2)
plt.show()

