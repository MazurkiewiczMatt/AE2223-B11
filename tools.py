from scipy.signal import find_peaks
import numpy as np
from math import pi, asin, sin

def range_calc(PSD, L, freq, chirp_time, phi_1, phi_2):
    
    PSD2 = PSD[L[0]:L[-1]]  # called PSD2 and not PSD not to mess up with outside values 
    freq2 = freq[L[0]:L[-1]] # same reasoning as above 
    indices = find_peaks(PSD2)[0]

    freq_peaks = [] 
    phase1_peaks = []  # is not a peak, but is the phase correponding to the frequency of a peak
    phase2_peaks = [] 
    for i in range(len(indices)):
        index = indices[i]
        freq_peaks.append(freq2[index])
        phase1_peaks.append(phi_1[i])
        phase2_peaks.append(phi_2[i])

    B = 250e6  # Hz (bandwidth range0)
    c = 2.99792458e8  # m/s
    T = chirp_time * 16 #total time
    #d = 16E-3 # m (distance between the two receivers)
    d_test_2 = (sin((pi/180) * (76/2)))**-1 * (0.0125 / 2)
    f_temp = 24E9
    range_lst = []
    geo_angle_lst = [] #test list 
    velocity_lst = []

    for k in range(len(freq_peaks)):
        try:
            range_lst = np.append(range_lst, c * T * freq_peaks[k] / (2 * B))
            omega = phase2_peaks[k] - phase1_peaks[k]  # difference between phases 
            if abs(omega) > pi:
                omega = 2*pi - omega
            sine = c * omega / (2 * pi * f_temp * d_test_2)
            assert (-1 <= sine <= 1), "Can't calculate asin of value outside of <-1, 1>."
            geo_angle_lst = np.append(geo_angle_lst, (180 / pi)*asin(sine))
            velocity_lst = np.append(velocity_lst, (c * abs(omega) / (4 * pi * f_temp * T)))
        except Exception as err:
            print("\033[;1m" + "Error: " + "\033[0;0m" + str(err))
            geo_angle_lst = np.append(geo_angle_lst, 0)
            velocity_lst = np.append(velocity_lst, 0)

    return range_lst, freq_peaks, geo_angle_lst

def fourier(chirps, t, realim, duration):
    dt = duration / len(chirps[realim])  # realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t)

    f_hat = np.fft.fft(chirps[realim], n)  # already zero padded 
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude  (np.real returns + 0*j)
    
    # Filtering:
    threshold = max(PSD)  # Find the highest peak
    indices = PSD > threshold / 2 # high-pass filter based on largest signal
    f_hat = f_hat * indices # Cleaned signal
    PSD = PSD * indices # Cleaned signal

    phase = np.angle(f_hat)
    freq = (1 / (dt * n)) * np.arange(n)  # Hz
    L = np.arange(1, np.floor(n / 2), dtype='int')  #u dont need two "mirrors"
    return PSD, freq, L, phase

def chirp_func(timestamp, radar_msg):
    rx1_re = np.array(radar_msg[int(timestamp)].data_rx1_re)
    rx1_im = np.array(radar_msg[int(timestamp)].data_rx1_im)
    rx2_re = np.array(radar_msg[int(timestamp)].data_rx2_re)
    rx2_im = np.array(radar_msg[int(timestamp)].data_rx2_im)

    # The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
    # and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).

    y = [rx1_re, rx1_im, rx2_re, rx2_im]
    no_chirps = radar_msg[int(timestamp)].dimx  #16
    length_chirp = radar_msg[int(timestamp)].dimy #128 
    chirps_temp = [] #intialize list 
    
    for i in range(no_chirps):  # Each i is one chirp. i ranges from 0 up to and including no_chirps - 1.
        temp_lst = []   
        for j in y:  # Each j is one type of measurement.
            temp_lst.append(
                j[length_chirp * i:length_chirp * (i + 1)])  # Add data that corresponds to current chirp
        chirps_temp.append(temp_lst)
    chirps_temp = np.array(chirps_temp)

    final_list = [[], [], [], []]  
    for i in range(128): #Average 16 chirps to 1 chirp for one timestamp 
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
    chirps = np.array(final_list)
    
    final_list = [[], [], [], []]  
# this could be probably done neater with list comprehension, like 
# example = [item for item in question]
    '''for i in range(128): #Average 16 chirps to 1 chirp for one timestamp 
        avg_calc_rx1re = np.array([])
        avg_calc_rx1im = np.array([])
        avg_calc_rx2re = np.array([])
        avg_calc_rx2im = np.array([])
        for k in range(16):
            final_list[0].append(np.average(np.append(avg_calc_rx1re, chirps_temp[k][0][i])))
            final_list[1].append(np.average(np.append(avg_calc_rx1im, chirps_temp[k][1][i])))
            final_list[2].append(np.average(np.append(avg_calc_rx2re, chirps_temp[k][2][i])))
            final_list[3].append(np.average(np.append(avg_calc_rx2im, chirps_temp[k][3][i])))   
    chirps = np.array(final_list)'''
    return chirps, no_chirps, length_chirp 