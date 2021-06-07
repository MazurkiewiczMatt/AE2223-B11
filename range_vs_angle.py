import rosbag
import numpy as np
import math
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tools import fourier, chirp_func, phase_calc, range_angle_velocity_calc, combined_FFT, PSD_calc, real_distance, real_angle, find_nearest_peak, get_file, get_folder_file
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
ori_x = [] #x-axis orientation
ori_y = [] #y-axis orientation
ori_z = [] #z-axis orientation
ori_w = [] #collective axis rotation

bagnumber = 41   # minimum 1, maximum 100
directory = get_folder_file('Bags', str(bagnumber) + '.bag')
with rosbag.Bag(directory) as bag: #Open the specific file to analyse 
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

chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg, 0)

# Optitrack obstacle data
directory = get_folder_file('Bags', 'trial_overview.csv')
obstacle_data = pd.read_csv(directory)

# --------------------------------- PROCESS DATA --------------------------------
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  #Total time in seconds
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

# Calculate range, geometrical angle and velocity
range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)

# Focus on only the closest object
range1, angle1, velocity1 = find_nearest_peak(FFT_RX1_combined, range_temp1, geo_angle_lst1, velocity_lst1)

# ----------------- Optitrack and Obstacle processing ---------------------------

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

#Distance and angel fucntions of drone relative to obctacle(s)
distance_to_obstacle = real_distance(x_drone,y_drone, obstacle_x, obstacle_z)
angle_to_obstacle, drone_yaw = real_angle(x_drone, y_drone, obstacle_x, obstacle_z, ox_drone, oy_drone, oz_drone, ow_drone)

# --------------- PLOT DATA ---------------------
font = {'family' : 'DejaVu Sans',
        'size'   : 15}

plt.rc('font', **font)

fig = plt.figure() #intialise figure
fig2 = plt.figure() # second frame
fig3 = plt.figure() # Isolate FFT magnitude plot to save it more easily
fig4 = plt.figure() # Isolate the radial plot to save it more easily

#Plot 1
ax1 = fig.add_subplot(1,1,1) 
ax1.set_ylabel('Magnitude')
ax1.set_xlabel('Normalised frequency')
ax1.set_title('Radar - FFT magnitude of chirp 1')

#Plot 2
ax2 = fig3.add_subplot(1,1,1) 
ax2.set_ylabel('Phase [rad]')
ax2.set_xlabel('Normalised frequency')
ax2.set_title('Radar - FFT phase of chirp 1')

#Plot 3
ax3 = fig4.add_subplot(1,1,1, projection='polar')
ax3.set_ylabel('Range [m]')
ax3.set_xlabel('Angle [deg]') #maybe?
ax3.set_title('Radar - Range vs Geometric angle (polar)')

#Plot 4
ax4 = fig2.add_subplot(1,1,1)
ax4.set_ylabel('Y [m]')
ax4.set_xlabel('X [m]')
ax4.set_title('Optitrack - Top View (Cartesian)')

#Plot 3 adjustments to convert to polar plot
ax3.set_rorigin(0)
ax3.set_theta_zero_location('N', offset=0) #Theta zero set to face north 
ax3.set_theta_direction(-1) #Flip direction of angle 
ax3.set_thetamin(-45) #FOV limit from drone sensors
ax3.set_thetamax(45) #FOV limit
ax3.set_rlim(0, 10) #Max detectable range should be 10 meters or less

#Plot 1 adjustments
ax1.set_ylim(0, 2500) #PSD values get very large in magnitude

#Plot 4 adjustments
ax4.set_ylim(-5, 5) #Area is 10 by 10 and centered around the middle
ax4.set_xlim(-5, 5) #Area enclosed

#Allow for plots to be updated by slider
p2, = ax3.plot(angle1, range1, 'o')
p9, = ax3.plot(angle_to_obstacle, distance_to_obstacle, 'o', color='r')

p3, = ax1.plot(freq_RX1, PSD_RX1, label='RX1')
p4, = ax1.plot(freq_RX2, PSD_RX2, label='RX2')
ax1.legend()

p5, = ax2.plot(freq_RX1, FFT_RX1_phase)
p6, = ax2.plot(freq_RX2, FFT_RX2_phase)

#Slider plot update
p7, = ax4.plot(x_drone, y_drone,'o') #drone position on axis
obstacle = plt.Circle((obstacle_x, obstacle_z), 0.2, color='r') #creates the obstacle as a circle on plot
ax4.add_patch(obstacle) #obstacle postion added to graph


# Plot text
txt_d = "Distance = " + "{:.3f}".format(distance_to_obstacle) + " Angle = " + "{:.3f}".format(np.degrees(angle_to_obstacle))
dist_text = ax4.text(0.04, 0.9, '', transform=ax4.transAxes)
dist_text.set_text(txt_d)

txt_r = "Distance = " + "{:.3f}".format(range1) + " Angle = " + "{:.3f}".format(np.degrees(angle1*-1))
radar_txt = ax3.text(0.04, 0.95, '', transform=ax3.transAxes)
radar_txt.set_text(txt_r)

# Plot line in wich direction the drone is pointing
line, = ax4.plot([], [], '-', lw=1)
thisx = [x_drone, x_drone - math.sin(drone_yaw)]
thisy = [y_drone, y_drone + math.cos(drone_yaw)]

line.set_data(thisx, thisy)

# Slider
ax_slide = plt.axes([0.25, 0.02, 0.65, 0.03])
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=(len(radar_msg) - 2), valinit=0, valstep=1)
# Val max is messages in terms of index, so start from 0

def update(val):
    current_v = s_factor.val  # Get current value on the slider
    timestamp = int(current_v)

    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg, 0)
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
        
    range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time, 1)
    
    # Focus on only the closest object
    range1, angle1, velocity1 = find_nearest_peak(FFT_RX1_combined, range_temp1, geo_angle_lst1, velocity_lst1)
    
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
    
    # Renew heading
    thisx = [x_drone, x_drone - math.sin(drone_yaw)]
    thisy = [y_drone, y_drone + math.cos(drone_yaw)]
    line.set_data(thisx, thisy)

    angle_to_obstacle_deg = np.degrees(angle_to_obstacle)
    if angle_to_obstacle_deg > 180:
        angle_to_obstacle_deg = -360 + angle_to_obstacle_deg

    # Plot all data
    p2.set_xdata(angle1*-1) 
    p2.set_ydata(range1) 

    p9.set_xdata(angle_to_obstacle) 
    p9.set_ydata(distance_to_obstacle)

    p3.set_xdata(freq_RX1)
    p3.set_ydata(PSD_RX1)

    p4.set_xdata(freq_RX2)
    p4.set_ydata(PSD_RX2)

    p5.set_xdata(freq_RX1)
    p5.set_ydata(FFT_RX1_phase)

    p6.set_xdata(freq_RX2)
    p6.set_ydata(FFT_RX2_phase)

    p7.set_xdata(x_drone)
    p7.set_ydata(y_drone)

    # Renew text
    # Text for optitrack
    txt_d = "Distance = " + "{:.2f}".format(distance_to_obstacle) + " Angle = " + "{:.2f}".format(angle_to_obstacle_deg)
    dist_text.set_text(txt_d)

    # Text for radar
    txt_r = "Distance = " + "{:.3f}".format(range1) + " Angle = " + "{:.3f}".format(np.degrees(angle1 *-1))
    radar_txt.set_text(txt_r)

    
    fig.canvas.draw()
    fig2.canvas.draw()
    fig3.canvas.draw()
    fig4.canvas.draw()

# Calling the slider function
s_factor.on_changed(update)
plt.show()