from scipy.signal import find_peaks
import numpy as np
from math import pi, asin, sin

def range_calc(PSD, L, freq, chirp_time, phi_1, phi_2):
    
    PSD2 = PSD[L[0]:L[-1]]
    freq2 = freq[L[0]:L[-1]]
    indices = find_peaks(PSD2)[0]
    #freq_peaks = [freq2[i] if freq2[i]<260 for i in indices]
    #phase1_peaks = [freq2[i] if phi_1[i]<260 for i in indices]
    #phase2_peaks = [freq2[i] if phi_2[i]<260 for i in indices]
    # TO DO: fix this and implement the shorter version
    #0
    freq_peaks = []
    phase1_peaks = []  # is not a peak, but is the phase correponding to the frequency of a peak
    phase2_peaks = []
    for i in range(len(indices)):
        index = indices[i]
        freq_peaks.append(freq2[index])
        phase1_peaks.append(phi_1[i])
        phase2_peaks.append(phi_2[i])
    #print('indices: ' + str(indices))
    print('\n\nfreq_peaks: ' + str(freq_peaks))

    freq_peaks_intermediate = []
    phase1_peaks_intermediate = []
    phase2_peaks_intermediate = []
    for i in range(len(freq_peaks)):
        freq_intermediate = freq_peaks[i]
        phase1_intermediate = phase1_peaks[i]
        phase2_intermediate = phase2_peaks[i]

        if freq_intermediate < 260:
            freq_peaks_intermediate.append(freq_intermediate)
            phase1_peaks_intermediate.append(phase1_intermediate)
            phase2_peaks_intermediate.append(phase2_intermediate)


    freq_peaks = freq_peaks_intermediate
    phase1_peaks = phase1_peaks_intermediate
    phase2_peaks = phase2_peaks_intermediate
    
    B = 250e6  # Hz (bandwidth range0)
    c = 2.99792458e8  # m/s
    T = chirp_time  # not sure about the 16? 
    #omega = abs(phi_2 - phi_1) # Phase difference [radians]
    #d = 16E-3 # m (distance between the two receivers)
    #d_test = 6.44E-3
    d_test_2 = (sin((pi/180) * (76/2)) * 2)**-1 * 0.0125 
    f_temp = 24E9
    range_lst = []
    geo_angle_lst = [] #test list 
    velocity_lst = []

    for k in range(len(freq_peaks)):
        try:
            range_lst = np.append(range_lst, c * T * freq_peaks[k] / (2 * B))
            omega = phase2_peaks[k] - phase1_peaks[k]
            print(omega)
            assert omega < pi, "Omega is not smaller than Pi."
            sine = c * omega / (2 * pi * f_temp * d_test_2)
            assert (-1 <= sine <= 1), "Can't calculate asin of value outside of <-1, 1>."
            geo_angle_lst = np.append(geo_angle_lst, (180 / pi)*asin(sine))
            velocity_lst = np.append(velocity_lst, (c * abs(omega) / (4 * pi * f_temp * T)))
        #geo_angle_lst = np.append(geo_angle_lst, (180 / pi )*asin((c * (phase2_peaks[k] - phase1_peaks[k]) / (2 * pi * f_temp * d))))
        #velocity_lst = np.append(velocity_lst, (c * abs(phase2_peaks[k] - phase1_peaks[k]) / (4 * pi * f_temp * T)))
        except Exception as err:
            print("\033[;1m" + "Error: " + "\033[0;0m" + str(err))
            geo_angle_lst = np.append(geo_angle_lst, 0)
            velocity_lst = np.append(velocity_lst, 0)

        
    print(str(range_lst) + ' meters')
    print(str(geo_angle_lst) + ' degrees')
    print(str(velocity_lst) + ' m/s')
    # Range 10m corresponds to 261 Hz. Probably not? 
    return range_lst, freq_peaks, geo_angle_lst


def fourier(chirps, t, realim, duration):  # from the internet
    dt = duration / len(chirps[realim])  # realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t)
    # Function
    f_hat = np.fft.fft(chirps[realim], n)  # already zero padded according to np documentation
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude sqrt(re^2 + im^2) type of thing. Do np.real bc it returns + 0*j.
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
    no_chirps = radar_msg[int(timestamp)].dimx #16 
    length_chirp = radar_msg[int(timestamp)].dimy #128 (when u print or run the code u see it)
    chirps_temp = [] #intialize list 
    for i in range(no_chirps):  # Each i is one chirp. i ranges from 0 up to and including no_chirps - 1.
        temp_lst = []  # temporary list to organise the chirps list properly
        for j in y:  # Each j is one type of measurement.
            temp_lst.append(
                j[length_chirp * i:length_chirp * (i + 1)])  # Add the data that correspond to the current chirp
        chirps_temp.append(temp_lst)
    chirps_temp = np.array(chirps_temp)
    # Take the average of all 16 chirps of one message
    #print(chirps_temp)
    #temp_list1 = np.array([])  # rx1re
    #temp_list2 = np.array([])  # rx1im
    #temp_list3 = np.array([])  # rx2re
    #temp_list4 = np.array([])  # rx2im

    final_list = [[], [], [], []]    # i is the type (rx1re etc), j is the value, len.128
    for i in range(128): #Combine 16 chirps to 1 chirp for one timestamp
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
        
    #print(final_list, len(final_list), len(final_list[0]), "we should get 4 128 HERWEEEEEE")
    # should be: longlist, 4, 128
    
    #print(temp_list1, len(temp_list1), "hhhhhhhhhhhhhheeeeeeeeeeeeeeeeeeee")
    

    chirps = np.array(final_list)
    return chirps, no_chirps, length_chirp