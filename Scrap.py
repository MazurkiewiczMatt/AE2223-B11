import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt

# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'
events = []
imu = []
optitrack = []
radar_time = []
radar_msg = []

with rosbag.Bag('1.bag') as bag:
    print(bag)
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

print(radar_msg[0])
# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval)
rx1_re = np.array(radar_msg[0].data_rx1_re)
rx1_im = np.array(radar_msg[0].data_rx1_im)
rx2_re = np.array(radar_msg[0].data_rx2_re)
rx2_im = np.array(radar_msg[0].data_rx2_im)

# The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
# and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).
y = [rx1_re, rx1_im, rx2_re, rx2_im]
no_chirps = radar_msg[0].dimx
length_chirp = radar_msg[0].dimy
chirps = []
for i in range(no_chirps):  # Each i is one chirp. i ranges from 0 up to and including no_chirps - 1.
    temp_lst = []  # temporary list to organise the chirps list properly
    for j in y:   # Each j is one type of measurement.
        temp_lst.append(j[length_chirp*i:length_chirp*(i+1)])   # Add the data that correspond to the current chirp
    chirps.append(temp_lst)

# --------------------------------- PROCESS DATA --------------------------------
# Another way of getting the duration would be either '12.1 / 128' or 'radar_time[1] - radar_time[0]'.
# These are not the same but I don't know what the difference is.
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = int(str(radar_msg[1].ts - radar_msg[0].ts))/1e9  # seconds. Originally is type 'genpy.rostime.Duration'.
chirp_time = duration / no_chirps
t = np.linspace(0, chirp_time, len(chirps[0][0]))    # x-axis [seconds]
amplitude = np.sqrt(chirps[0][0] ** 2 + chirps[0][1] ** 2)
'''for i in range(len(y1)):
    amplitude = math.sqrt((y1_im[i])2+(y1[i])2)
    phase = math.atan2(y1_im[i],y1[i])
    amp_tx_1.append(amplitude)
    phase_tx_1.append(phase)'''

B = 250e6   # Hz
c = 2.99792458e8  # m/s
T = duration
freq = 80    # Hz
R = c * T * freq / (2 * B)
print(str(R) + '  meters')

# ---------------------------------- PLOT DATA -----------------------------------

plt.suptitle('RADAR DATA')
plt.subplot(1, 2, 1)
plt.plot(t, amplitude)
plt.title('rx1_re (raw)')
plt.xlabel('Time [s]')
plt.ylabel('no idea')

n_zeros = 50
t = np.linspace(0, chirp_time, len(chirps[0][0]) + n_zeros)    # x-axis [seconds]
amplitude = np.append(amplitude, np.zeros(n_zeros))
dt = duration / len(amplitude)
n = len(t)

f_hat = np.fft.fft(amplitude, n) #already zero padded according to np documentation 
PSD = f_hat * np.conj(f_hat) / n
freq = (1/(dt*n)) * np.arange(n)   # Hz
L = np.arange(1, np.floor(n/2), dtype='int')

plt.subplot(1, 2, 2)
plt.plot(freq[L], PSD[L], 1)
plt.title('rx1_re (transformed)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Density [no idea]')
plt.show()

