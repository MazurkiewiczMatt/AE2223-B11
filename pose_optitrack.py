import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter


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


data = pd.read_csv('C:/Users/ic120/PycharmProjects/AE2223-B11/1/optitrack-pose.csv')

# input coordinates, in RHS
x_coordinates = data['pose.position.x']
y_coordinates = data['pose.position.y']
z_coordinates = data['pose.position.z']

timestamps = data['Time']

# input rotations in quaternion system
x_orientation = data['pose.orientation.x']
y_orientation = data['pose.orientation.y']
z_orientation = data['pose.orientation.z']
w_orientation = data['pose.orientation.w']

# never mind all this since this is because i thought to apply the matrices when i shouldn't have haha ha ha :')
# position = np.array([x_coordinates, z_coordinates, y_coordinates])
# new_position = np.empty(position.shape)
#
# for i in range(len(x_coordinates)):
#     Q = [w_orientation[i], x_orientation[i], y_orientation[i], z_orientation[i]]  # define input for function of
#     # transforming quaternion to rotation matrix
#     new_position[:, i] = quaternion_rotation_matrix(Q) @ position[:, i]
#
# new_data_dict = {'new position x': new_position[0, :], 'new position y': new_position[1, :],
#                'new position z': new_position[2, :]}  # i am really unsure about the y and z in here being like this
# # and not switched around because of the right hand and left hand coordinate systems
# data_new = pd.DataFrame(new_data_dict)
#
# x = data_new['new position x']
# y = data_new['new position y']
# z = data_new['new position z']

# basic filtering to make the line  more even:
n = 50
bb = [1.0 / n] * n
aa = 1
x = x_coordinates
y = y_coordinates
z = z_coordinates

xfilter = lfilter(bb, aa, x)
yfilter = lfilter(bb, aa, y)
zfilter = lfilter(bb, aa, z)

fig3, ax = plt.subplots()
fig3.suptitle('3D Trajectory', fontsize=20)
ax = plt.axes(projection='3d')
ax.plot3D(xfilter, yfilter, zfilter, linewidth='4')

plt.show()

