import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt


def d_to_float(drt):
    # converts genpy.rostime.Duration to a float
    return drt.secs + drt.nsecs / 1e9


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

timestamp = 0   # each timestamp is a chirp. Can be used to see what happens over time. Note: in the code there is
# timestamp +1, so if we want to see the last chirp, we need to change the code a bit.

print(radar_msg[timestamp])
# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval)
rx1_re = np.array(radar_msg[timestamp].data_rx1_re)
rx1_im = np.array(radar_msg[timestamp].data_rx1_im)
rx2_re = np.array(radar_msg[timestamp].data_rx2_re)
rx2_im = np.array(radar_msg[timestamp].data_rx2_im)

# The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
# and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).
y = [rx1_re, rx1_im, rx2_re, rx2_im]
no_chirps = radar_msg[timestamp].dimx
length_chirp = radar_msg[timestamp].dimy
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
duration = d_to_float(radar_msg[timestamp+1].ts - radar_msg[timestamp].ts)  # seconds. Originally is type 'genpy.rostime.Duration'.
chirp_time = duration / no_chirps
t = np.linspace(0, chirp_time, len(chirps[0][0]))    # x-axis [seconds]
amplitude = np.sqrt(chirps[0][0] ** 2 + chirps[0][1] ** 2)
'''for i in range(len(y1)):
    amplitude = math.sqrt((y1_im[i])2+(y1[i])2)
    phase = math.atan2(y1_im[i],y1[i])
    amp_tx_1.append(amplitude)
    phase_tx_1.append(phase)'''


# ---------------------------------- PLOT DATA -----------------------------------

plt.suptitle('RADAR DATA')
plt.subplot(1, 2, 1)
plt.plot(t, amplitude)
plt.title('rx1_re (raw)')
plt.xlabel('Time [s]')
plt.ylabel('no idea')

dt = duration / len(amplitude)
n = len(t)

f_hat = np.fft.fft(amplitude, n)   #already zero padded according to np documentation
PSD = np.real(f_hat * np.conj(f_hat) / n)
freq = (1/(dt*n)) * np.arange(n)   # Hz
L = np.arange(1, np.floor(n/2), dtype='int')  # I think the 1 excludes the first data point

print('\n\n\n\n' + str(PSD))
print(freq)

plt.subplot(1, 2, 2)
plt.plot(freq[L], PSD[L], marker='o')
plt.title('rx1_re (transformed)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Density [no idea]')

# ------------------------------------- CALCULATIONS -------------------------------------------
# Step 1: find the peaks (use closest peak?)
# Method: find the maximum point in the data, divide it by 2 and use that as a threshold to detect other peaks.
# Then select the peak with the lowest frequency (x-axis). If we want multi-target detection, skip this step.
PSD2 = PSD[L[0]:L[-1]]
peak_index = np.argmax(PSD2)   # match x interval with the graph. peak is the largest peak
peak_value = PSD2[peak_index]
threshold = peak_value / 3
plt.hlines(threshold, 0, freq[L[-1]], colors='orange')
indices = PSD2 > threshold  # these are also valid for the list 'freq'
freq2 = freq[L[0]:L[-1]]
freq_peaks = []
index_numbers = []
for i in range(len(indices)):
    if indices[i]:
        freq_peaks.append(freq2[i])
        index_numbers.append(i)

for j in range(len(index_numbers) - 1):
    print(j)
    print('freq2 ' + str(freq2[index_numbers[j]]))
    print('freq2 j+1   ' + str(freq2[index_numbers[j+1]]))

    if PSD2[index_numbers[j]] < PSD2[index_numbers[j+1]]:   # if two nodes next to each other are peaks, then this is one peak.
        indices[index_numbers[j]] = False
    else:
        indices[index_numbers[j+1]] = False

freq_peaks = []   # reset the peaks and construct it again
for i in range(len(indices)):
    if indices[i]:
        freq_peaks.append(freq2[i])

print('indices: ' + str(indices))
print('freq_peaks: ' + str(freq_peaks))

B = 250e6   # Hz
c = 2.99792458e8  # m/s
T = duration
range_lst = []
for k in freq_peaks:
    range_lst.append(c * T * k / (2 * B))
print(str(range_lst) + ' meters')
plt.show()
