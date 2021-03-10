import rosbag
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------- IMPORT BAG -------------------------------
bag = rosbag.Bag('1.bag')
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'
events = []
imu = []
optitrack = []
radar_time = []
radar_msg = []
for topic, msg, t in bag.read_messages(topics=['/radar/data']):
    radar_time.append(t)
    radar_msg.append(msg)
bag.close()

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

# --------------------------------- PLOT DATA --------------------------------
# Another way of getting the duration would be either '12.1 / 128' or 'radar_time[1] - radar_time[0]'.
# These are not the same but I don't know what the difference is.
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = int(str(radar_msg[1].ts - radar_msg[0].ts))/1e9  # seconds. Originally is type 'genpy.rostime.Duration'.
chirp_time = duration / no_chirps
t = np.linspace(0, chirp_time, len(chirps[0][0]))    # x-axis [seconds]
y = chirps[0][0]

plt.suptitle('RADAR DATA')
plt.subplot(1, 2, 1)
plt.plot(t, y)
plt.title('rx1_re (raw)')
plt.xlabel('Time [s]')
plt.ylabel('no idea')

dt = duration / len(y)
n = len(t)

f_hat = np.fft.fft(y, n)
PSD = f_hat * np.conj(f_hat) / n
freq = (1/(dt*n)) * np.arange(n)   # Hz
L = np.arange(1, np.floor(n/2), dtype='int')

plt.subplot(1, 2, 2)
plt.plot(freq[L], PSD[L], 1)
plt.title('rx1_re (transformed)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Density [no idea]')
plt.show()

