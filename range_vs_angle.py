import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from tools import range_calc, fourier, chirp_func

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files

radar_time = []
radar_msg = []
with rosbag.Bag('1.bag') as bag:
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

timestamp = 0  # each timestamp has a message. Can be used to see what happens over time.

# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval)
d
# Chirps: We have a list of 16 other lists, each sublist 4 subsublists (rx1re,rx1im,rx2re rx2img), 
# in that subsublist 128 values for each. (real and complex)
chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds
chirp_time = duration / no_chirps #time for one chirp
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

# Want to do the Fourier transform over receiver 1 and 2, therefore the process is done twice
# Fourier transform
# First, do the FFT separately over the real and imaginary signals
PSD_1re, freq_1re, L_1re, phase_1re, f_hat_1re = fourier(chirps, t, 0, duration) # rx_1red
PSD_1im, freq_1im, L_1im, phase_1im, f_hat_1im = fourier(chirps, t, 1, duration) # rx_1im
PSD_2re, freq_2re, L_2re, phase_2re, f_hat_2re = fourier(chirps, t, 2, duration) # rx_2re
PSD_2im, freq_2im, L_2im, phase_2im, f_hat_2im = fourier(chirps, t, 3, duration) # rx_2im

# FFT of the combined (complex) signal = combination of the outputs of the FFT.
# F[a+jb] = F[a] + jF[b]

FFT_RX_1_total_re = f_hat_1re.real - f_hat_1im.imaginary
FFT_RX_1_total_im = f_hat_1im.imaginary - f_hat_2im.real

FFT_RX_2_total_re = f_hat_2re.real - f_hat_2im.imaginary
FFT_RX_2_total_im = f_hat_2re.imaginary- f_hat_2im.real

phase1 = np.arctan(FFT_RX_1_total_im / FFT_RX_1_total_re)
phase2 = np.arctan(FFT_RX_2_total_im / FFT_RX_2_total_re)

# Calculate the range and geometric angle
range_temp1, freq_peaks1, geo_angle_lst1 = range_calc(PSD, L, freq, chirp_time, phase1, phase2)

range_temp2, freq_peaks2, geo_angle_lst2 = range_calc(PSD, L, freq, chirp_time, phase1, phase2)

# ---------------------------------- PLOT DATA -----------------------------------
fig, (ax1, ax2) = plt.subplots(1,2)

PSD, freq, L, _ = fourier(chirps, t, 2, duration)
p2, = ax1.plot(freq[L], PSD[L], label='RX2_re')
plt.ylim(-10, 10)

PSD, freq, L, _ = fourier(chirps, t, 0, duration)
p, = ax1.plot(freq[L], PSD[L], label='RX1_re')
plt.ylim(-10, 10)

plt.legend()

# Need to make geo_angle_lst and range_temp always have the same dimension (each timestamp). Just add the appropriate number of zeroes.
while len(geo_angle_lst2) < 20:
    geo_angle_lst2 = np.append(geo_angle_lst2, 0)
    range_temp1 = np.append(range_temp1, 0)
p3, = ax2.plot(geo_angle_lst2, range_temp1, 'o', label='obstacle')
plt.xlim(-38, 38)
plt.ylim(0, 10)

# slider
ax_slide = plt.axes([0.25, 0.02, 0.65, 0.03])
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=(len(radar_msg) - 2), valinit=0, valstep=1)
# val max is messages in terms of index, so start from 0
# list of ranges
range_drone = []

# update the data 
def update(val):
    current_v = s_factor.val #get current value on the slider
    timestamp = int(current_v)
    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

    # Number in the input of function is: 0=rx1_real; 1=rx1_imag; 2=rx1_real; 3=rx2_imag
    PSD, freq, L, phase1 = fourier(chirps, t, 0, duration) # rx_1 real
    p.set_ydata(PSD[L])

    PSD, freq, L, phase2 = fourier(chirps, t, 2, duration) # rx_2 real
    p2.set_ydata(PSD[L]) 

    # Range and angle calculation for most recent transformed data (PSD, freq, L)
    range_temp1, freq_peaks1, geo_angle_lst1 = range_calc(PSD, L, freq, chirp_time, phase1, phase2)

    while len(geo_angle_lst1) < 20:
        geo_angle_lst1 = np.append(geo_angle_lst1, 0)
        range_temp1 = np.append(range_temp1, 0)

    p3.set_ydata(range_temp1)
    fig.canvas.draw()

# calling the slider function
s_factor.on_changed(update)
plt.show()