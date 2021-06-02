import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from tools import fourier, chirp_func, phase_calc, range_angle_velocity_calc, combined_FFT, PSD_calc, real_distance, real_angle, find_nearest_peak, reject_outliers, get_file, get_folder_file
import pandas as pd
from scipy.stats import norm

error_list_distance = []
error_list_angle = []
error_list_velocity = []

for bagnumber in range(1, 101):
    # Bags 15, 68, 93 are invalid tests, so are left out.
    if bagnumber != 15 and bagnumber != 68 and bagnumber != 93:
        print(bagnumber)
        # ---------------------------- IMPORT BAG -------------------------------
        # All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files
        #List initialisation
        radar_time = [] #radar time  - Unnesecary comment
        radar_msg = [] #radar message - Unnesecary comment
        #Optitrack
        opti_x = [] #x-axis drone movement data
        opti_y = [] #y-axis drone  - Unnesecary comment
        opti_z = [] #z-axis drone - Unnesecary comment
        opti_t = [] #Position data time stamps
        #Rotation data
        ori_x = [] #x-axis orientation
        ori_y = [] #y-axis orientation - Unnesecary comment
        ori_z = [] #z-axis orientation - Unnesecary comment
        ori_w = [] #collective axis rotation

        bagnumber = bagnumber   # minimum 1, maximum 100  - Unnesecary comment
        directory = get_folder_file('Bags', str(bagnumber) + '.bag')
        with rosbag.Bag(directory) as bag: #Open the specific file to analyse 
            for topic, msg, t in bag.read_messages(topics=['/radar/data']): #Organise data for radar from Topics
                radar_time.append(t) #time data - Unnesecary comment
                radar_msg.append(msg) #radar data - Unnesecary comment

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
        chirps_for_velocity, _, _ = chirp_func(timestamp, radar_msg, 1)

        # Optitrack data
        directory = get_folder_file('Bags', 'trial_overview.csv')
        obstacle_data = pd.read_csv(directory)

        # --------------------------------- PROCESS DATA --------------------------------
        total_time = (radar_msg[timestamp-1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9
        duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  #Total time of one message in seconds
        chirp_time = duration / no_chirps #Time for 1 chirp (16 chirps total)
        t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds] - x-axis of what?
        msg_rate = 1 / (chirp_time * no_chirps) #The frequency at which messages are taken
        sample_rate = msg_rate * len(chirps[0]) #Samplig frequency for chirp data (128 = len(chirps[0]))

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

        # Same for the other chirp used for velocity calculations
        f_hat_1re_vel = fourier(chirps_for_velocity, t, 0, duration)  # rx_1re - Unnesecary comment
        f_hat_1im_vel = fourier(chirps_for_velocity, t, 1, duration)  # rx_1im - Unnesecary comment
        f_hat_2re_vel = fourier(chirps_for_velocity, t, 2, duration)  # rx_2re - Unnesecary comment
        f_hat_2im_vel = fourier(chirps_for_velocity, t, 3, duration)  # rx_2im - Unnesecary comment

        FFT_RX1_combined_vel = combined_FFT(f_hat_1re_vel, f_hat_1im_vel) 
        FFT_RX2_combined_vel = combined_FFT(f_hat_2re_vel, f_hat_2im_vel)

        PSD_RX1_vel, freq_RX1_vel, FFT_RX1_combined_vel = PSD_calc(FFT_RX1_combined_vel, t, duration, chirps_for_velocity, sample_rate)
        PSD_RX2_vel, freq_RX2_vel, FFT_RX2_combined_vel = PSD_calc(FFT_RX2_combined_vel, t, duration, chirps_for_velocity, sample_rate)

        # Same for the other chirp used for velocity calculations
        FFT_RX1_phase_vel = phase_calc(FFT_RX1_combined_vel) 
        FFT_RX2_phase_vel = phase_calc(FFT_RX2_combined_vel)

        # Range, geometrical angle and velocity calculation
        range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time, phi_velocity=FFT_RX1_phase_vel)

        # Focus on only the closest object
        range1, angle1, velocity1 = find_nearest_peak(FFT_RX1_combined, range_temp1, geo_angle_lst1, velocity_lst1)

        # ----------------- Optitrack and Obstacle processing ---------------------------
        # input column coordinates (Obstacle), in RHS coordinates
        x_column_tot = obstacle_data["Obstacle x"]  # Need the full column for later - Unnesecary comment
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
        velocity_radar_time = []
        velocity_optitrack_time = []
        counter = 0

        for timestamp in range(1, len(radar_msg)-2):
            
            chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg, 0)
            chirps_for_velocity, _, _ = chirp_func(timestamp, radar_msg, 1)

            duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
            chirp_time = duration / no_chirps
            t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds] - x-axis for what?

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

            # Same for the other chirp used for velocity calculations
            f_hat_1re_vel = fourier(chirps_for_velocity, t, 0, duration)  # rx_1re
            f_hat_1im_vel = fourier(chirps_for_velocity, t, 1, duration)  # rx_1im
            f_hat_2re_vel = fourier(chirps_for_velocity, t, 2, duration)  # rx_2re
            f_hat_2im_vel = fourier(chirps_for_velocity, t, 3, duration)  # rx_2im

            FFT_RX1_combined_vel = combined_FFT(f_hat_1re_vel, f_hat_1im_vel) 
            FFT_RX2_combined_vel = combined_FFT(f_hat_2re_vel, f_hat_2im_vel)

            PSD_RX1_vel, freq_RX1_vel, FFT_RX1_combined_vel = PSD_calc(FFT_RX1_combined_vel, t, duration, chirps_for_velocity, sample_rate)
            PSD_RX2_vel, freq_RX2_vel, FFT_RX2_combined_vel = PSD_calc(FFT_RX2_combined_vel, t, duration, chirps_for_velocity, sample_rate)

            FFT_RX1_phase_vel = phase_calc(FFT_RX1_combined_vel) 
            FFT_RX2_phase_vel = phase_calc(FFT_RX2_combined_vel)

            # Range, geometrical angle and velocity calculations
            range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time, phi_velocity=FFT_RX1_phase_vel)
            
            # Focus on only the closest object
            range1, angle1, velocity1 = find_nearest_peak(FFT_RX1_combined, range_temp1, geo_angle_lst1, velocity_lst1)
            
            # ----------------- Optitrack and Obstacle processing ---------------------------
            # Renew optitrack data
            x_drone = opti_x[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
            x_drone_next = opti_x[int(((timestamp + 1) * len(opti_x)/(len(radar_msg) - 2))-1)]
            y_drone = opti_y[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
            y_drone_next = opti_y[int(((timestamp + 1) * len(opti_x)/(len(radar_msg) - 2))-1)]

            # Renew optitrack heading
            ox_drone = ori_x[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
            oy_drone = ori_y[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
            oz_drone = ori_z[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]
            ow_drone = ori_w[int((timestamp * len(opti_x)/(len(radar_msg) - 2))-1)]

            # Real distance and angle to obstacle and drone orientation in optitrack coordinates
            distance_to_obstacle = real_distance(x_drone, y_drone, obstacle_x, obstacle_z)
            distance_to_obstacle_next = real_distance(x_drone_next, y_drone_next, obstacle_x, obstacle_z)
            angle_to_obstacle, drone_yaw = real_angle(x_drone, y_drone, obstacle_x, obstacle_z, ox_drone, oy_drone, oz_drone, ow_drone)
            
            # Make angle work
            angle_to_obstacle_deg = np.degrees(angle_to_obstacle)
            if angle_to_obstacle_deg > 180:
                angle_to_obstacle_deg = -360 + angle_to_obstacle_deg

            # Velocity by optitrack
            velocity_temp = (distance_to_obstacle - distance_to_obstacle_next) / duration

            # Renew heading
            range_time.append(range1)
            optitrack_range_time.append(distance_to_obstacle)
            angle_time.append(angle1*-1)
            optitrack_angle_time.append(angle_to_obstacle_deg)
            velocity_radar_time.append(velocity1)
            velocity_optitrack_time.append(velocity_temp)

        # - - - - - - - - - - - PLOT - - - - - - - - - - - - - - - - - -
        #t1 = np.linspace(0, total_time, len(range_time))  - Unnesecary comment
        range_radar = np.array(range_time) 
        range_opti = np.array(optitrack_range_time)
        angle_radar = np.array(angle_time)
        angle_opti = np.array(optitrack_angle_time)
        velocity_radar = np.array(velocity_radar_time)
        velocity_opti = np.array(velocity_optitrack_time)


        # Error plots
        # First filter out the data where the obstacle is behind the drone
        indices = abs(angle_opti) < 38
        angle_opti = angle_opti[indices]
        angle_radar = angle_radar[indices]
        range_radar = range_radar[indices]
        range_opti = range_opti[indices]
        velocity_opti = velocity_opti[indices]
        velocity_radar = velocity_radar[indices]
        
        try:
            error_distance_percent = np.abs(range_radar - range_opti) * 100 / np.abs(range_opti)
            error_angle_percent = np.abs(angle_opti - angle_radar*180/np.pi)
            error_velocity_percent = np.abs(velocity_opti - velocity_radar)
        except Exception as e:
            print("Exception ecountered: " + str(e))
            error_distance_percent = 0
            error_angle_percent = 0
            error_velocity_percent = 0
            print("The values changed to 0")

        # These are the error representations for the whole bag
        error_distance_tot = np.mean(error_distance_percent) 
        error_angle_tot = np.mean(error_angle_percent)
        error_velocity_tot = np.mean(error_velocity_percent)

        error_list_distance.append(error_distance_tot)
        error_list_angle.append(error_angle_tot)
        error_list_velocity.append(error_velocity_tot)


error_distance_plot_values = np.array(error_list_distance)
error_angle_plot_values = np.array(error_list_angle)
error_velocity_plot_values = np.array(error_list_velocity)
x_bags = np.arange(1, 101, 1)
# Reject invalid bags.
for val in x_bags:
    if val == 15 or val == 68 or val == 93:
        x_bags = np.delete(x_bags, val-1)
print(x_bags)

print("\n\n\nDISTANCE \n")
print(error_distance_plot_values)

print("\n\n\nANGLE \n")
print(error_angle_plot_values)

print("\n\n\nVELOCITY \n")
print(error_velocity_plot_values)

# Find mean, median and interquartile range for the errors
print("MEAN:")
print("Distance error: " + "{:.4f}".format(np.mean(error_distance_plot_values)) + " %")
print("Angle error: " + "{:.4f}".format(np.mean(error_angle_plot_values)) + " deg")
print("Velocity error: " + "{:.4f}".format(np.mean(error_velocity_plot_values)) + " m/s")

print("MEDIAN:")
print("Distance error: " + "{:.4f}".format(np.median(error_distance_plot_values)) + " %")
print("Angle error: " + "{:.4f}".format(np.median(error_angle_plot_values)) + " deg")
print("Velocity error: " + "{:.4f}".format(np.median(error_velocity_plot_values)) + " m/s")

print("IQR:")
print("Distance error: " + "{:.4f}".format(np.percentile(error_distance_plot_values, 25)) + "-" + "{:.4f}".format(np.percentile(error_distance_plot_values, 75)) + " %")
print("Angle error: " + "{:.4f}".format(np.percentile(error_angle_plot_values, 25)) + "-" + "{:.4f}".format(np.percentile(error_angle_plot_values, 75)) + " deg")
print("Velocity error: " + "{:.4f}".format(np.percentile(error_velocity_plot_values, 25)) + "-" + "{:.4f}".format(np.percentile(error_velocity_plot_values, 75)) + " m/s")

# Normal plots
fig = plt.figure()

fig6 = plt.figure()
fig7 = plt.figure()
fig8 = plt.figure()

# Plot 1
ax1 = fig6.add_subplot(1,1,1)
ax1.set_title('Distance error all bags')
ax1.scatter(x_bags,error_distance_plot_values)
ax1.axhline(np.mean(error_distance_plot_values), color='r')
ax1.set_xlabel('Bag number')
ax1.set_ylabel('Range error [%]')

# Plot 2 
ax2 = fig7.add_subplot(1,1,1)
ax2.set_title('Angle error all bags')
ax2.scatter(x_bags,error_angle_plot_values)
ax2.axhline(np.mean(error_angle_plot_values), color='r')
ax2.set_xlabel('Bag number')
ax2.set_ylabel('Angle error [deg]')


# Plot 3
ax3 = fig8.add_subplot(1,1,1)
ax3.set_title('Velocity error all bags')
ax3.scatter(x_bags, error_velocity_plot_values, label='Radar')
ax3.axhline(np.mean(error_velocity_plot_values), color='r')
ax3.set_xlabel('Bag number')
ax3.set_ylabel('Velocity error [m/s]')

# Boxplots
#fig2 = plt.figure()

fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()

# Plot 1
ax4 = fig3.add_subplot(1,1,1)
ax4.set_title('Distance error all bags')
ax4.set_ylabel('Range error [%]')
ax4.boxplot(error_distance_plot_values)


# Plot 2
ax5 = fig4.add_subplot(1,1,1)
ax5.set_title('Angle error all bags')
ax5.set_ylabel('Angle error [deg]')
ax5.boxplot(error_angle_plot_values)


# Plot 3
ax6 = fig5.add_subplot(1,1,1)
ax6.set_title('Velocity error all bags')
ax6.set_ylabel('Velocity error [m/s]')
ax6.boxplot(error_velocity_plot_values)

plt.show()
