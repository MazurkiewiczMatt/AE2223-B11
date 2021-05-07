import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from tools import fourier, chirp_func, phase_calc, range_angle_velocity_calc, combined_FFT, PSD_calc, check

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files

radar_time = []
radar_msg = []
with rosbag.Bag('1.bag') as bag:
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

timestamp = 0  # Each timestamp has a message. Can be used to see what happens over time.

# ---------------------------------- LOAD DATA --------------------------------
"""
This is the data from a single datapoint (time interval) 
Chirps: We have a list of 16 other lists, each sublist 4 subsublists (rx1re,rx1im,rx2re rx2img), 
in that subsublist 128 values for each. (real and complex)
"""

chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------

duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds
chirp_time = duration / no_chirps 
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

f_hat_1re = fourier(chirps, t, 0, duration)  # rx_1re
f_hat_1im = fourier(chirps, t, 1, duration)  # rx_1im
f_hat_2re = fourier(chirps, t, 2, duration)  # rx_2re
f_hat_2im = fourier(chirps, t, 3, duration)  # rx_2im

# FFT of the combined (complex) signal = combination of the outputs of the FFT. Calculation is as follows:
FFT_RX1_combined = combined_FFT(f_hat_1re, f_hat_1im)
FFT_RX2_combined = combined_FFT(f_hat_2re, f_hat_2im)

PSD_RX1, freq_RX1, FFT_RX1_combined = PSD_calc(FFT_RX1_combined, t, duration, chirps)
PSD_RX2, freq_RX2, FFT_RX2_combined = PSD_calc(FFT_RX2_combined, t, duration, chirps)

FFT_RX1_combined, FFT_RX2_combined = check(FFT_RX1_combined, FFT_RX2_combined)

# Calculate angle of the complex numbers.
FFT_RX1_phase = phase_calc(FFT_RX1_combined) 
FFT_RX2_phase = phase_calc(FFT_RX2_combined) 

range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)
# --------------- PLOT DATA ---------------------

range_time = []
for timestamp in range(len(radar_msg)-2):
    print(timestamp)
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

    PSD_RX1, freq_RX1, FFT_RX1_combined = PSD_calc(FFT_RX1_combined, t, duration, chirps)
    PSD_RX2, freq_RX2, FFT_RX2_combined = PSD_calc(FFT_RX2_combined, t, duration, chirps)

    FFT_RX1_combined, FFT_RX2_combined = check(FFT_RX1_combined, FFT_RX2_combined)

    # Calculate angle of the complex numbers
    FFT_RX1_phase = phase_calc(FFT_RX1_combined)
    FFT_RX2_phase = phase_calc(FFT_RX2_combined)
        
    range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)

    range_temp1, geo_angle_lst1 = check(range_temp1, geo_angle_lst1)
    range_time.append(range_temp1)

print(range_time)
t1 = np.array([])
y1 = np.array([])
for i in range(len(range_time)):
    for j in range(len(range_time[i])):
        t1 = np.append(t1, i)
        y1 = np.append(y1, range_time[i][j])

plt.scatter(t1, y1)
plt.xlabel('message number ~ time')
plt.ylabel('range [m]')
plt.show()