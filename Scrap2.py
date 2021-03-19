import rosbag
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def range_calc(PSD, L, freq):
    PSD2 = PSD[L[0]:L[-1]]
    peak_index = np.argmax(PSD2)  # match x interval with the graph. peak is the largest peak
    peak_value = PSD2[peak_index]
    threshold = peak_value / 10
    # plt.hlines(threshold, 0, freq[L[-1]], colors='orange')
    indices = PSD2 > threshold  # these are also valid for the list 'freq'
    freq2 = freq[L[0]:L[-1]]
    index_numbers = []

    index_numbers = [i for i, value in enumerate(indices) if value]  # get indices of true values in indices list

    for j in range(len(index_numbers) - 1):
        # print(j)
        # print('freq2 ' + str(freq2[index_numbers[j]]))
        # print('freq2 j+1   ' + str(freq2[index_numbers[j]+1]))

        if PSD2[index_numbers[j]] < PSD2[
            index_numbers[j] + 1]:  # if two nodes next to each other are peaks, then this is one peak.
            indices[index_numbers[j]] = False
        else:
            indices[index_numbers[j] + 1] = False

    freq_peaks = []
    for i in range(len(indices)):
        if indices[i]:
            freq_peaks.append(freq2[i])

    # print('indices: ' + str(indices))
    print('freq_peaks: ' + str(freq_peaks))

    B = 250e6  # Hz
    c = 2.99792458e8  # m/s
    T = duration
    range_lst = []
    for k in freq_peaks:
        range_lst.append(c * T * k / (2 * B))
    print(str(range_lst) + ' meters')
    return range_lst


# ---------------------------- IMPORT BAG -------------------------------
# All topics are: '/dvs/events', '/dvs/imu', '/optitrack/pose', '/radar/data'
events = []
imu = []
optitrack = []
radar_time = []
radar_msg = []
with rosbag.Bag('1.bag') as bag:
    # print(bag)
    for topic, msg, t in bag.read_messages(topics=['/radar/data']):
        radar_time.append(t)
        radar_msg.append(msg)

timestamp = 0  # each timestamp is a message. Can be used to see what happens over time. Note: in the code there is


# timestamp +1, so if we want to see the last message, we need to change the code a bit.

# print(radar_msg[timestamp])


# ---------------------------------- LOAD DATA --------------------------------
# This is the data from a single datapoint (time interval)
def chirp_func(timestamp):
    rx1_re = np.array(radar_msg[int(timestamp)].data_rx1_re)
    rx1_im = np.array(radar_msg[int(timestamp)].data_rx1_im)
    rx2_re = np.array(radar_msg[int(timestamp)].data_rx2_re)
    rx2_im = np.array(radar_msg[int(timestamp)].data_rx2_im)

    # The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
    # and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).
    y = [rx1_re, rx1_im, rx2_re, rx2_im]
    no_chirps = radar_msg[int(timestamp)].dimx
    length_chirp = radar_msg[int(timestamp)].dimy
    chirps_temp = []
    for i in range(no_chirps):  # Each i is one chirp. i ranges from 0 up to and including no_chirps - 1.
        temp_lst = []  # temporary list to organise the chirps list properly
        for j in y:  # Each j is one type of measurement.
            temp_lst.append(
                j[length_chirp * i:length_chirp * (i + 1)])  # Add the data that correspond to the current chirp
        chirps_temp.append(temp_lst)
    chirps_temp = np.array(chirps_temp)
    # Take the average of all 16 chirps of one message
    # print(chirps_temp)
    # temp_list1 = np.array([])  # rx1re
    # temp_list2 = np.array([])  # rx1im
    # temp_list3 = np.array([])  # rx2re
    # temp_list4 = np.array([])  # rx2im

    final_list = [[], [], [], []]  # i is the type (rx1re etc), j is the value, len.128
    for i in range(128):
        avg_calc_rx1re = np.array([])
        avg_calc_rx1im = np.array([])
        avg_calc_rx2re = np.array([])
        avg_calc_rx2im = np.array([])

        for k in range(16):
            avg_calc_rx1re = np.append(avg_calc_rx1re, chirps_temp[k][0][i])
            avg_calc_rx1im = np.append(avg_calc_rx1im, chirps_temp[k][1][i])
            avg_calc_rx2re = np.append(avg_calc_rx2re, chirps_temp[k][2][i])
            avg_calc_rx2im = np.append(avg_calc_rx2im, chirps_temp[k][3][i])

        final_val_rx1re = np.average(avg_calc_rx1re)
        final_val_rx1im = np.average(avg_calc_rx1im)
        final_val_rx2re = np.average(avg_calc_rx2re)
        final_val_rx2im = np.average(avg_calc_rx2im)

        final_list[0].append(final_val_rx1re)
        final_list[1].append(final_val_rx1im)
        final_list[2].append(final_val_rx2re)
        final_list[3].append(final_val_rx2im)

    # print(final_list, len(final_list), len(final_list[0]), "we should get 4 128 HERWEEEEEE")
    # sould be: longlist, 4, 128

    # print(temp_list1, len(temp_list1), "hhhhhhhhhhhhhheeeeeeeeeeeeeeeeeeee")

    chirps = np.array(final_list)
    return chirps, no_chirps, length_chirp


# Chirps: We have a list, in that list 16 other lists, in these lists we have 4 lists (rx1re,rx1im,rx2re rx2img), in that list 128 values for each.

chirps, no_chirps, length_chirp = chirp_func(timestamp)

# --------------------------------- PROCESS DATA --------------------------------
# Another way of getting the duration would be either '12.1 / 128' or 'radar_time[1] - radar_time[0]'.
# These are not the same but I don't know what the difference is.
# 'duration' is the time of one message. 'chirp_time' is the duration of one chirp.
duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
chirp_time = duration / no_chirps
t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
y1 = chirps[0]  # real
y2 = chirps[2]  # imaginary
'''for i in range(len(y1)):
    amplitude = math.sqrt((y1_im[i])2+(y1[i])2)
    phase = math.atan2(y1_im[i],y1[i])
    amp_tx_1.append(amplitude)
    phase_tx_1.append(phase)'''


# ---------------------------------- PLOT DATA -----------------------------------
# fucntion
# plt.suptitle('RADAR DATA')
# plt.subplot(1, 2, 1)
# plt.plot(t, chirps[0])
# plt.plot(t, chirps[1])
# plt.title('rx1_re (raw)')
# plt.xlabel('Time [s]')
# plt.ylabel('no idea')

def fourier(chirps, t, realim):
    dt = duration / len(chirps[realim])  # realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t)
    # Function
    f_hat = np.fft.fft(chirps[realim], n)  # already zero padded according to np documentation
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude
    freq = (1 / (dt * n)) * np.arange(n)  # Hz
    L = np.arange(1, np.floor(n / 2), dtype='int')  # I think the 1 excludes the first data point
    return PSD, freq, L


fig = plt.figure()
ax1 = fig.subplots()

PSD, freq, L = fourier(chirps, t, 2)
# plt.yscale("log")
plt.ylim(0, 1000)
p2, = ax1.plot(freq[L], PSD[L], label='RX2_re')

PSD, freq, L = fourier(chirps, t, 0)
# plt.yscale("log")
plt.ylim(0, 1000)
p, = ax1.plot(freq[L], PSD[L], label='RX1_re')

plt.legend()

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
s_factor = Slider(ax_slide, 'Time', valmin=0, valmax=186, valinit=0, valstep=1)

# list of ranges
range_drone = []


# update
def update(val):
    current_v = s_factor.val
    timestamp = int(current_v)
    chirps, no_chirps, length_chirp = chirp_func(timestamp)
    duration = (radar_msg[timestamp + 1].ts - radar_msg[timestamp].ts).to_nsec() / 1e9  # seconds.
    chirp_time = duration / no_chirps
    t = np.linspace(0, chirp_time, len(chirps[0]))  # x-axis [seconds]
    y1 = chirps[0]  # real
    y2 = chirps[2]  # real rx2
    # print(t[1], y1[1], timestamp, "Still Running")

    dt = duration / len(chirps[0])
    n = len(t)
    # Function
    PSD, freq, L = fourier(chirps, t, 0)
    p.set_ydata(PSD[L])

    PSD, freq, L = fourier(chirps, t, 1)
    p2.set_ydata(PSD[L])

    range_temp = range_calc(PSD, L, freq)
    range_drone.append(range_temp)
    print('range drone....', range_drone)

    fig.canvas.draw()
    # fig2.canvas.draw()


# calling function
s_factor.on_changed(update)

# ------------------------------------- CALCULATIONS -------------------------------------------
# Step 1: find the peaks (use closest peak?)
# Method: find the maximum point in the data, divide it by 2 and use that as a threshold to detect other peaks.
# Then select the peak with the lowest frequency (x-axis). If we want multi-target detection, skip this step.
plt.show()