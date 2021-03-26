import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from tools import range_calc, fourier, chirp_func
# SLIDER
# SLIDER
# SLIDER
# SLIDER this has a slider
# SLIDER
# SLIDER

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'
events = []
imu = []
optitrack = []
radar_time = []
radar_msg = []
with rosbag.Bag('1.bag') as bag:
    #print(bag)
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

timestamp = 0  # each timestamp is a message. Can be used to see what happens over time. Note: in the code there is
# timestamp +1, so if we want to see the last message, we need to change the code a bit.

#print(radar_msg[timestamp])


# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval)

# Chirps: We have a list, in that list 16 other lists, in these lists we have 4 lists (rx1re,rx1im,rx2re rx2img), in that list 128 values for each. (real and complex stuff)
# the line above is your point of reference (the line is also reffered to as the sacred text) 
chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------
# Another way of getting the duration would be either '12.1 / 128' or 'radar_time[1] - radar_time[0]'.
# These are not the same but I don't know what the difference is.
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
chirp_time = duration / no_chirps #time for one chirp
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
y1 = chirps[0]  # realrx_1 from line 134 its the list 
y2 = chirps[2]  # realrx_2
'''for i in range(len(y1)):
    amplitude = math.sqrt((y1_im[i])2+(y1[i])2)
    phase = math.atan2(y1_im[i],y1[i])
    amp_tx_1.append(amplitude)
    phase_tx_1.append(phase)'''

# Define new axis system for range and angle plot
dt = duration / len(chirps[0])
n = len(t)
# Function
PSD, freq, L, phase1 = fourier(chirps, t, 0, duration) # added phase and y is now phase data
PSD, freq, L, phase2 = fourier(chirps, t, 2, duration) # rx_2 rea
range_temp, freq_peaks, geo_angle_lst = range_calc(PSD, L, freq, chirp_time, phase1, phase2)

# ---------------------------------- PLOT DATA -----------------------------------
# fucntion
# plt.suptitle('RADAR DATA')
# plt.subplot(1, 2, 1)
# plt.plot(t, chirps[0])
# plt.plot(t, chirps[1])
# plt.title('rx1_re (raw)')
# plt.xlabel('Time [s]')
# plt.ylabel('no idea')


fig = plt.figure()
ax1 = fig.subplots()


PSD, freq, L, _ = fourier(chirps, t, 2, duration)
#plt.yscale("log")
plt.ylim(-10, 10)
p2, = ax1.plot(freq[L], PSD[L], label='RX2_re')

PSD, freq, L, _ = fourier(chirps, t, 0, duration)
#plt.yscale("log")
plt.ylim(-10, 10)
p, = ax1.plot(freq[L], PSD[L], label='RX1_re')

#_, freq_peaks = range_calc(PSD, L, freq, chirp_time)

#p3, = ax1.plot(freq_peaks, np.linspace(50, 51, len(freq_peaks)), 'o')

plt.legend()


fig = plt.figure()
ax2 = fig.subplots()
p3, = ax2.plot(geo_angle_lst, range_temp, 'o', label='obstacle')
plt.xlim(-38, 38)
plt.ylim(0, 10)
'''range_fig = plt.figure()
ax2 = fig.subplot()
p3, = ax2.plot()#25 meters larget range anyway'''

# plt.subplot(1, 2, 2)
'''fig = plt.figure()
ax = fig.subplots()
plt.subplots_adjust(bottom = 0.25)
p = ax.plot(freq[L], PSD[L], marker='o')
plt.title('rx1_re (transformed)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Density [no idea]')'''

# slider
ax_slide = plt.axes([0.25, 0.02, 0.65, 0.03])
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=(len(radar_msg) - 2), valinit=0, valstep=1)
# val max is messages in terms of index so start from 0
# list of ranges
range_drone = []

#Almost heaven
#west viginia
#blue ridge mountains
#shenendoha riveeer
#life is old therE
#older than teh trees
# #younger than the mountains
#blowin like a breeze
#COUNTRY ROOOOOAAADS TAKE ME HOOOOOOOME
#TO THE PLAAINS 
#WHERE I BELLOOOOOOOONG

# update
def update(val):
    current_v = s_factor.val #get current value on the slider
    timestamp = int(current_v)
    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
    y1 = chirps[0]  # real rx1 no update basically if not here
    y2 = chirps[2]  # real rx2
    #print(t[1], y1[1], timestamp, "Still Running")

    dt = duration / len(chirps[0])
    n = len(t)
    # Function
    PSD, freq, L, phase1 = fourier(chirps, t, 0, duration) # added phase and y is now phase data
    p.set_ydata(phase1[L])

    PSD, freq, L, phase2 = fourier(chirps, t, 2, duration) # rx_2 rea
    p2.set_ydata(phase2[L]) 

    range_temp, freq_peaks, geo_angle_lst = range_calc(PSD, L, freq, chirp_time, phase1, phase2)
    p3.set_ydata(range_temp)
    #p3, = ax1.plot(freq_peaks, np.linspace(50, 51, len(freq_peaks)), 'o')
    #range_drone.append(range_temp)
    #print('range drone....', range_drone)

    #p3.set_xdata(freq_peaks)
    fig.canvas.draw()
    # fig2.canvas.draw()


# calling function
s_factor.on_changed(update)

# ------------------------------------- CALCULATIONS -------------------------------------------
# Step 1: find the peaks (use closest peak?)
# Method: find the maximum point in the data, divide it by 2 and use that as a threshold to detect other peaks.
# Then select the peak with the lowest frequency (x-axis). If we want multi-target detection, skip this step.
plt.show()