import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from tools import fourier, chirp_func, phase_calc, range_angle_velocity_calc, combined_FFT, PSD_calc, real_distance, real_angle, find_nearest_peak
import pandas as pd

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files
#List initialisation
radar_time = [] #radar time
radar_msg = [] #radar message
#Optitrack
opti_x = [] #x-axis drone movement data
opti_y = [] #y-axis drone
opti_z = [] #z-axis drone
opti_t = [] #Position data time stamps
#Rotation data
ori_x = [] #x-axis oreintation
ori_y = [] #y-axis orientation
ori_z = [] #z-axis orientation
ori_w = [] #collective axis rotation

bagnumber = 22   # minimum 1, maximum 100
with rosbag.Bag(str(bagnumber) + '.bag') as bag: #Open the specific file to analyse 
    for topic, msg, t in bag.read_messages(topics=['/radar/data']): #Organise data for radar from Topics
        radar_time.append(t) #time data
        radar_msg.append(msg) #radar data

    for topic, msg, t in bag.read_messages(topics=['/optitrack/pose']): #Organise data for Optitrack from the same Topics
        opti_t.append(t) #time data
        #Drone position and Orientation data 
        opti_x.append(msg.pose.position.x)
        opti_y.append(msg.pose.position.z)
        opti_z.append(msg.pose.position.y)
        ori_x.append(msg.pose.orientation.x)
        ori_y.append(msg.pose.orientation.y)
        ori_z.append(msg.pose.orientation.z)
        ori_w.append(msg.pose.orientation.w)
#Change lists to arrays for Numpy compatiblity
#Drone Position
opti_x = np.array(opti_x)*-1
opti_y = np.array(opti_y)
opti_z = np.array(opti_z)
opti_t = np.array(opti_t)
#Drone Orientation
ori_x = np.array(ori_x)
ori_y = np.array(ori_y)
ori_z = np.array(ori_z)
ori_w = np.array(ori_w)

timestamp = 0  # Each timestamp has a message. Is used to see what happens over time.
# ---------------------------------- LOAD DATA --------------------------------
"""
This is the data from a single datapoint (time interval) 
Chirps: We have a list of 16 other lists, each sublist 4 subsublists (rx1re,rx1im,rx2re rx2img), 
in that subsublist 128 values for each. (real and complex)
"""

chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------
total_time = (radar_msg[timestamp-1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  #Total time of one message in seconds
chirp_time = duration / no_chirps #Time for 1 chirp (16 chirps total)
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
msg_rate = 1 / (chirp_time * no_chirps) #The frequency at which messages are taken
sample_rate = msg_rate * len(chirps[0]) #Samlig frequency for chirp data (128 = len(chirps[0]))

f_hat_1re = fourier(chirps, t, 0, duration)  # rx_1re
f_hat_1im = fourier(chirps, t, 1, duration)  # rx_1im
f_hat_2re = fourier(chirps, t, 2, duration)  # rx_2re
f_hat_2im = fourier(chirps, t, 3, duration)  # rx_2im

# FFT of the combined (complex) signal = combination of the outputs of the FFT. Calculation is as follows:
#Check tools.py for more details on functions 
FFT_RX1_combined = combined_FFT(f_hat_1re, f_hat_1im) 
FFT_RX2_combined = combined_FFT(f_hat_2re, f_hat_2im)

PSD_RX1, freq_RX1, FFT_RX1_combined = PSD_calc(FFT_RX1_combined, t, duration, chirps, sample_rate)
PSD_RX2, freq_RX2, FFT_RX2_combined = PSD_calc(FFT_RX2_combined, t, duration, chirps, sample_rate)

# Calculate angle of the complex numbers.
FFT_RX1_phase = phase_calc(FFT_RX1_combined) 
FFT_RX2_phase = phase_calc(FFT_RX2_combined) 

range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)

# Focus on only the closest object
range1, angle1 = find_nearest_peak(6, FFT_RX1_combined, range_temp1, geo_angle_lst1)

# Optitrack data
obstacle_data = pd.read_csv(r'C:\Users\frank\Documents\TU Delft\Year 2\Q3\Project\Github\trial_overview.csv')

# input column coordinates (Obstacle), in RHS coordinates
x_column_tot = obstacle_data["Obstacle x"]  # Need the full column for later
x_column = x_column_tot.drop_duplicates() #Remove repeated values
y_column = obstacle_data["Obstacle y"]
y_column = y_column.drop_duplicates() 
z_column = obstacle_data["Obstacle z"]
z_column = z_column.drop_duplicates()

# column coordinates in single array, x-y-z for the 3 variants
obstacles = np.stack((x_column, y_column, z_column), axis=1) # xyz: for coordinates, use x for x center and z for y center  

# Array of obstacle positions per bag file.
x_value = x_column_tot[bagnumber-1]
for i in range(3): #3 cooridnates present
    if x_value == obstacles[i][0]: #If numbers match then flip the x cooridnates to adjust graph
        obstacle_x, obstacle_y, obstacle_z = obstacles[i]
        obstacle_x *= -1

# Drone 2D position data in each timestamp
x_drone = opti_x[0] 
y_drone = opti_y[0]
# Drone orientation data in each timestamp
ox_drone = ori_x[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)] #obtain drone data with index adjustments 
oy_drone = ori_y[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
oz_drone = ori_z[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
ow_drone = ori_w[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]

#Distance and Angle fucntions of drone relative to obctacle(s)
distance_to_obstacle = real_distance(x_drone,y_drone, obstacle_x, obstacle_z)
angle_to_obstacle, drone_yaw = real_angle(x_drone, y_drone, obstacle_x, obstacle_z, ox_drone, oy_drone, oz_drone, ow_drone)


thisx = [x_drone, x_drone - math.sin(drone_yaw)]
thisy = [y_drone, y_drone + math.cos(drone_yaw)]

# Create lists to store values
range_time = []
optitrack_range_time = []
angle_time = []
optitrack_angle_time = []
counter = 0

for timestamp in range(len(radar_msg)-2):
    counter += 1
    print(counter)
    
    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

    f_hat_1re = fourier(chirps, t, 0, duration)  # rx_1re
    f_hat_1im = fourier(chirps, t, 1, duration)  # rx_1im
    f_hat_2re = fourier(chirps, t, 2, duration)  # rx_2re
    f_hat_2im = fourier(chirps, t, 3, duration)  # rx_2im
    
    FFT_RX1_combined = combined_FFT(f_hat_1re, f_hat_1im)
    FFT_RX2_combined = combined_FFT(f_hat_2re, f_hat_2im)

    PSD_RX1, freq_RX1, FFT_RX1_combined = PSD_calc(FFT_RX1_combined, t, duration, chirps, sample_rate)
    PSD_RX2, freq_RX2, FFT_RX2_combined = PSD_calc(FFT_RX2_combined, t, duration, chirps, sample_rate)

    # Calculate angle of the complex numbers
    FFT_RX1_phase = phase_calc(FFT_RX1_combined)
    FFT_RX2_phase = phase_calc(FFT_RX2_combined)
        
    range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)
    
    # Focus on only the closest object
    range1, angle1 = find_nearest_peak(6, FFT_RX1_combined, range_temp1, geo_angle_lst1)
    
    # Renew optitrack data
    x_drone = opti_x[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
    y_drone = opti_y[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]

    # Renew optitrack heading
    ox_drone = ori_x[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
    oy_drone = ori_y[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
    oz_drone = ori_z[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
    ow_drone = ori_w[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]

    # Real distance and angle to obstacle and drone orientation in optitrack coordinates
    distance_to_obstacle = real_distance(x_drone,y_drone,obstacle_x, obstacle_z)
    angle_to_obstacle, drone_yaw = real_angle(x_drone, y_drone, obstacle_x, obstacle_z, ox_drone, oy_drone, oz_drone, ow_drone)
    
    '''if angle_to_obstacle > np.pi:
        angle_to_obstacle = -2*np.pi + angle_to_obstacle'''
    angle_to_obstacle_deg = np.degrees(angle_to_obstacle)
    if angle_to_obstacle_deg > 180:
        angle_to_obstacle_deg = -360 + angle_to_obstacle_deg

    # Renew heading
    range_time.append(range1)
    optitrack_range_time.append(distance_to_obstacle)
    angle_time.append(angle1*-1)
    optitrack_angle_time.append(angle_to_obstacle_deg)

# - - - - - - - - - - - PLOT - - - - - - - - - - - - - - - - - -
t1 = np.linspace(0, total_time, len(range_time))
y1 = range_time
y2 = optitrack_range_time
y_radar = angle_time
y_opti = optitrack_angle_time

fig = plt.figure()
fig2 = plt.figure()


ax1 = fig.add_subplot(1,2,1)
ax1.scatter(t1, y1, label='Radar')
ax1.scatter(t1, y2, label='Optitrack')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('range [m]')
ax1.legend()

# found something
#angles_2pi = np.mod(angles, 2*np.pi)
'''y_radar = np.mod(y_radar, 2*np.pi)
y_opti = np.mod(y_opti, 2*np.pi)'''

ax2 = fig.add_subplot(1,2,2)
ax2.scatter(t1, np.array(y_radar)*180/np.pi, label='Radar')
ax2.scatter(t1, y_opti, label='Optitrack')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('angle [deg]')
ax2.legend()

# Error plots
'''


ax3 = fig2.add_subplot(1,1,1)

'''

plt.show()