import rosbag
import numpy as np
from matplotlib import pyplot as plt
# example change
# Hello this is Frank
bag = rosbag.Bag('1.bag')

#/dvs/events, '/dvs/imu', '/optitrack/pose', '/radar/data'
events = []
imu = []
optitrack = []
radar_time = []
radar_msg = []
for topic, msg, t in bag.read_messages(topics=['/radar/data']):
    radar_time.append(t)
    radar_msg.append(msg)
# msg.dimx     msg....
print(radar_msg[0])
bag.close()


y1_1 = np.array(radar_msg[0].data_rx1_re)
y1_2 = np.array(radar_msg[0].data_rx1_im)
y2_1 = np.array(radar_msg[0].data_rx2_re)
y2_2 = np.array(radar_msg[0].data_rx2_im)

y1 = []
for i in range(len(y1_1)):
    if i <= 128:
        y1.append(y1_1[i])
x1 = np.linspace(0, 12.1, len(y1))

y1_im = []
for i in range(len(y1_2)):
    if i <= 128:
        y1_im.append(y1_2[i])
x1_im = np.linspace(0, 12.1, len(y1_im))

y2 = []
for i in range(len(y2_1)):
    if i <= 128:
        y2.append(y2_1[i])
x2 = np.linspace(0, 12.1, len(y2))
# --------------------------------------------------------------
dt = 12.1 / len(y1)
n = len(x1)

f_hat = np.fft.fft(y1, n)
PSD = f_hat * np.conj(f_hat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

plt.suptitle('RADAR DATA')

plt.subplot(2, 4, 1)
plt.title('data_rx1_re (raw)')
plt.plot(x1, y1, 0)

plt.subplot(2, 4, 2)
plt.title("data_rx1_re (transformed)")
plt.plot(freq[L], PSD[L], 1)


dt = 12.1 / len(y2)
n = len(x2)

f_hat = np.fft.fft(y2, n)
PSD = f_hat * np.conj(f_hat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

plt.subplot(2, 4, 5)
plt.title('data_rx2_re (raw)')
plt.plot(x2, y2, 0)

plt.subplot(2, 4, 6)
plt.title("data_rx2_re (transformed)")
plt.plot(freq[L], PSD[L], 1)


dt = 12.1 / len(y1_im)
n = len(x1_im)

f_hat = np.fft.fft(y1_im, n)
PSD = f_hat * np.conj(f_hat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

plt.subplot(2, 4, 3)
plt.title('data_rx1_im (raw)')
plt.plot(x1_im, y1_im, 0)

plt.subplot(2, 4, 4)
plt.title("data_rx1_im (transformed)")
plt.plot(freq[L], PSD[L], 1)


plt.show()
