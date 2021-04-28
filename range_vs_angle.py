import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from tools import range_angle_calc, fourier, chirp_func, update, plot_graph_slider, phase_calc, PSD_calc, range_angle_velocity_calc2


# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'; can be accessed in folder 1 as cvs files

radar_time = []
radar_msg = []
with rosbag.Bag('9.bag') as bag:
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

timestamp = 0  # each timestamp has a message. Can be used to see what happens over time.

# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval) 
# Chirps: We have a list of 16 other lists, each sublist 4 subsublists (rx1re,rx1im,rx2re rx2img), 
# in that subsublist 128 values for each. (real and complex)
chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds
chirp_time = duration / no_chirps #time for one chirp
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

# We want to do the Fourier transform over receiver 1 and 2, therefore the process is done twice
# Fourier transform
# First, do the FFT separately over the real and imaginary signals
#BELOW IS THE FOURIER FUNCTION  
f_hat_1re = fourier(chirps, t, 0, duration) # rx_1re
f_hat_1im = fourier(chirps, t, 1, duration) # rx_1im
f_hat_2re = fourier(chirps, t, 2, duration) # rx_2re
f_hat_2im = fourier(chirps, t, 3, duration) # rx_2im

# FFT of the combined (complex) signal = combination of the outputs of the FFT. Calculation is as follows:
# F[a+jb] = F[a] + jF[b] = c + jd + je - f = (c-f) + j(d+e) 
# F[a] = c + jd 
# F[b] = e + jf

FFT_RX1_combined = (f_hat_1re.real - f_hat_1im.imag) + 1j * (f_hat_1re.imag + f_hat_1im.real) #fingers crossed, no hoping
FFT_RX2_combined = (f_hat_2re.real - f_hat_2im.imag) + 1j * (f_hat_2re.imag + f_hat_2im.real) #second reciever 
PSD_RX1, freq_RX1, L_RX1 = PSD_calc(FFT_RX1_combined, t, duration, chirps) # RX1
PSD_RX2, freq_RX2, L_RX2 = PSD_calc(FFT_RX2_combined, t, duration, chirps) # RX2

# Calculate angle of the complex numbers.
FFT_RX1_phase = phase_calc(FFT_RX1_combined) #Rx 1 phase
FFT_RX2_phase = phase_calc(FFT_RX2_combined) #Rx 2 phase

#phi_angle = (FFT_RX1_phase - FFT_RX2_phase) # difference between phases of the two Rx antennae

range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc2(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)
'''
plt.plot(geo_angle_lst1, range_temp1, 'o')
plt.xlim(-38, 38)
plt.ylim(0, 10)
plt.show()
'''
# --------------- PLOT DATA ---------------------
fig, (ax1) = plt.subplots(1)

p2, = ax1.plot(geo_angle_lst1, range_temp1, 'o')

plt.xlim(-38, 38)

# slider
ax_slide = plt.axes([0.25, 0.02, 0.65, 0.03])
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=(len(radar_msg) - 2), valinit=0, valstep=1)
# val max is messages in terms of index, so start from 0

# update the data 
def update(val):
    current_v = s_factor.val #get current value on the slider
    timestamp = int(current_v)

    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

    f_hat_1re = fourier(chirps, t, 0, duration) # rx_1re
    f_hat_1im = fourier(chirps, t, 1, duration) # rx_1im
    f_hat_2re = fourier(chirps, t, 2, duration) # rx_2re
    f_hat_2im = fourier(chirps, t, 3, duration) # rx_2im

    FFT_RX1_combined = (f_hat_1re.real - f_hat_1im.imag) + 1j * (f_hat_1re.imag + f_hat_1im.real) #fingers crossed, no hoping
    FFT_RX2_combined = (f_hat_2re.real - f_hat_2im.imag) + 1j * (f_hat_2re.imag + f_hat_2im.real) #second reciever 
    PSD_RX1, freq_RX1, L_RX1 = PSD_calc(FFT_RX1_combined, t, duration, chirps) # RX1
    PSD_RX2, freq_RX2, L_RX2 = PSD_calc(FFT_RX2_combined, t, duration, chirps) # RX2

    # Calculate angle of the complex numbers.
    FFT_RX1_phase = phase_calc(FFT_RX1_combined) #Rx 1 phase
    FFT_RX2_phase = phase_calc(FFT_RX2_combined) #Rx 2 phase
    
    #phi_angle = (FFT_RX1_phase - FFT_RX2_phase) # difference between phases of the two Rx antennae
    
    range_temp1, range_temp2, geo_angle_lst1, velocity_lst1 = range_angle_velocity_calc2(freq_RX1, freq_RX2, FFT_RX1_phase, FFT_RX2_phase, chirp_time)
    
    p2.set_xdata(geo_angle_lst1)
    p2.set_ydata(range_temp1)
    fig.canvas.draw()

# calling the slider function
s_factor.on_changed(update)
plt.show()