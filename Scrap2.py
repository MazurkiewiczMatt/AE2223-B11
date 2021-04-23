import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from tools import range_calc, fourier, chirp_func

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'
radar_time = []
radar_msg = []
with rosbag.Bag('1.bag') as bag:
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)
timestamp = 0  # Each timestamp is a message. Used to see what happens over time. 
#Note: In code there is timestamp +1, so for last message, we need to change the code a bit.

# ---------------------------------- LOAD DATA --------------------------------
# Data from a single datapoint (time interval)
# Chirps: 1 list, in that list 16 lists, in these 4 lists (rx1re,rx1im,rx2re rx2img), in each of these 128 values.
chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)

# --------------------------------- PROCESS DATA --------------------------------
# '12.1 / 128' or 'radar_time[1] - radar_time[0]' also gives duration (not the same)
# 'duration' -> time of one message. 
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
chirp_time = duration / no_chirps #duration of one chirp
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]

# ---------------------------------- PLOT DATA -----------------------------------
#All plotting use plt.yscale("log") for log scale
fig = plt.figure()
ax1 = fig.subplots()

PSD, freq, L, _ = fourier(chirps, t, 2, duration)
plt.ylim(0, 1000)
p2, = ax1.plot(freq[L], PSD[L], label='RX2_re')

PSD, freq, L, _ = fourier(chirps, t, 0, duration)
plt.ylim(0, 1000)
p, = ax1.plot(freq[L], PSD[L], label='RX1_re')

plt.legend()
range_drone = []

for i in range(186): #for entire duration of flight
    current_v = i
    timestamp = int(current_v)
    chirps, no_chirps, length_chirp = chirp_func(timestamp, radar_msg)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
    
    PSD, freq, L, phase1 = fourier(chirps, t, 0, duration)
    p.set_ydata(phase1[L])

    PSD, freq, L, phase2 = fourier(chirps, t, 1, duration)
    p2.set_ydata(phase2[L])

    range_temp, _, _ = range_calc(PSD, L, freq, chirp_time, phase1, phase2)
    range_drone.append(range_temp)
    
plt.show()

t1 = np.array([])
y1 = np.array([])
for i in range(len(range_drone)):
    for j in range(len(range_drone[i])):
        t1 = np.append(t1, i)
        y1 = np.append(y1, range_drone[i][j])

plt.scatter(t1, y1)
plt.xlabel('message number ~ time')
plt.ylabel('range [m]')
plt.show()