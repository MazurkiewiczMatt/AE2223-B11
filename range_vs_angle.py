import rosbag
import numpy as np
import math
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tools import fourier, chirp_func, phase_calc, range_angle_velocity_calc, combined_FFT, PSD_calc, check

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files

radar_time = []
radar_msg = []
with rosbag.Bag('10.bag') as bag:
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
msg_rate = 1 / (chirp_time * no_chirps)
sample_rate = msg_rate * len(chirps[0])

f_hat_1re = fourier(chirps, t, 0, duration)  # rx_1re
f_hat_1im = fourier(chirps, t, 1, duration)  # rx_1im
f_hat_2re = fourier(chirps, t, 2, duration)  # rx_2re
f_hat_2im = fourier(chirps, t, 3, duration)  # rx_2im

# FFT of the combined (complex) signal = combination of the outputs of the FFT. Calculation is as follows:
FFT_RX1_combined = combined_FFT(f_hat_1re, f_hat_1im)
FFT_RX2_combined = combined_FFT(f_hat_2re, f_hat_2im)

PSD_RX1, freq_RX1, FFT_RX1_combined = PSD_calc(FFT_RX1_combined, t, duration, chirps, sample_rate)
PSD_RX2, freq_RX2, FFT_RX2_combined = PSD_calc(FFT_RX2_combined, t, duration, chirps, sample_rate)

FFT_RX1_combined, FFT_RX2_combined = check(FFT_RX1_combined, FFT_RX2_combined)

# Calculate angle of the complex numbers.
FFT_RX1_phase = phase_calc(FFT_RX1_combined) 
FFT_RX2_phase = phase_calc(FFT_RX2_combined) 

range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)
# --------------- PLOT DATA ---------------------

fig = plt.figure()


ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,4, projection='polar')
ax4 = fig.add_subplot(2,2,3)

ax3.set_rorigin(0)
ax3.set_theta_zero_location('N', offset=0)
# ----------
ax3.set_theta_direction(-1)
# ----------
ax3.set_thetamin(-45)
ax3.set_thetamax(45)
ax3.set_rlim(0, 10)

ax1.set_xlim(0,0.5)
ax1.set_ylim(0, 2500)

'''
ax2.set_xlim()
ax2.set_ylim(0, 10)
'''
p2, = ax3.plot(geo_angle_lst1, range_temp1, 'o')

p3, = ax1.plot(freq_RX1, PSD_RX1)
p4, = ax1.plot(freq_RX2, PSD_RX2)

p5, = ax2.plot(freq_RX1, FFT_RX1_phase)
p6, = ax2.plot(freq_RX2, FFT_RX2_phase)

p7, = ax4.plot(np.degrees(geo_angle_lst1), range_temp1, 'o')

# Slider
ax_slide = plt.axes([0.25, 0.02, 0.65, 0.03])
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=(len(radar_msg) - 2), valinit=0, valstep=1)
# Val max is messages in terms of index, so start from 0

def update(val):
    current_v = s_factor.val  # Get current value on the slider
    timestamp = int(current_v)

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

    FFT_RX1_combined, FFT_RX2_combined = check(FFT_RX1_combined, FFT_RX2_combined)

    # Calculate angle of the complex numbers
    FFT_RX1_phase = phase_calc(FFT_RX1_combined)
    FFT_RX2_phase = phase_calc(FFT_RX2_combined)
        
    range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)

    range_temp1, geo_angle_lst1 = check(range_temp1, geo_angle_lst1)

    p2.set_xdata(geo_angle_lst1)
    p2.set_ydata(range_temp1)
    
    p3.set_xdata(freq_RX1)
    p3.set_ydata(PSD_RX1)

    p4.set_xdata(freq_RX2)
    p4.set_ydata(PSD_RX2)

    p5.set_xdata(freq_RX1)
    p5.set_ydata(FFT_RX1_phase)

    p6.set_xdata(freq_RX2)
    p6.set_ydata(FFT_RX2_phase)

    p7.set_xdata(np.degrees(geo_angle_lst1))
    p7.set_ydata(range_temp1)

    fig.canvas.draw()

# Calling the slider function
s_factor.on_changed(update)
plt.show()