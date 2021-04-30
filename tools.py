from scipy.signal import find_peaks
import numpy as np
from math import pi, asin, sin
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def range_angle_velocity_calc(freq1, freq2, phi_1, phi_2, chirp_time):
    B = 250e6   # Hz (bandwidth range)
    c = 2.99792458e8   # m/s
    T = chirp_time  # total time
    
    d_test_2 = (sin((pi/180) * (76/2)))**-1 * (0.0125 / 2)  # TODO: need update Distance between two receivers
    f_temp = 24E9  # TODO: need update
    range_lst1 = (c * T * freq1 / (2 * B) )  # Range formula for receiver 1
    range_lst2 = (c * T * freq2 / (2 * B) )  # Range formula for receiver 2
    
    delta_omega = phi_1 - phi_2  # Difference between phases 
    
    # NOTE: No Errors
    z = -1
    for i in delta_omega:
        z += 1
        if i <= 0:
            delta_omega[z] = i + 2*np.pi 
    
    delta_omega = delta_omega - np.pi
    # --------

    temp_constant = c * delta_omega / (2 * pi * f_temp * d_test_2)  # Angle formula 
    
    geo_angle_lst = np.arcsin(temp_constant)  # Final angle found in degrees
    velocity_lst = (c * abs(delta_omega) / (4 * pi * f_temp * T))  # Final velocity found including velocity formula

    return range_lst1, range_lst2, geo_angle_lst, velocity_lst

def phase_calc(FFT):  # NOTE: beautiful, outstanding, revolutionary
    return np.angle(FFT)  # imaginary over real based on definitions
    

def combined_FFT(f_hat_re, f_hat_im):
    # F[a+jb] = F[a] + jF[b] = c + jd + je - f = (c-f) + j(d+e) 
    # F[a] = c + jd 
    # F[b] = e + jf
    return (f_hat_re.real - f_hat_im.imag) + 1j * (f_hat_re.imag + f_hat_im.real) 

def fourier(chirps, t, realim, duration):  # Calculate the fourier of the signal 
    dt = duration / len(chirps[realim])  # Realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t)  # Total number of timestamps
    f_hat = np.fft.fft(chirps[realim], n)  # Frequency array already zero padded according to documentation
    return f_hat

def PSD_calc(f_hat, t, duration, chirps):
    """ Intensity of frequency calculated using fourier """
    dt = duration / len(chirps[0])  # Realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t)  # Total number of timestamps
    
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude  (np.real returns + 0*j)
    
    freq = (1 / (dt * n)) * np.arange(n)  # Frequency calculated
    L = np.arange(1, np.floor(n / 2), dtype='int')  # Prevent a divison of intensity over a twice long domain
    freq = freq[L]
    PSD = PSD[L]
    f_hat = f_hat[L]

    threshold = max(PSD)  # Find the highest peak
    indices = PSD > threshold / 2  # high-pass filter based on largest signal
    PSD = PSD * indices

    return PSD[PSD != 0], freq[PSD != 0], f_hat[PSD != 0]



def chirp_func(timestamp, radar_msg):
    # Average the chirps for faster computation
    rx1_re = np.array(radar_msg[int(timestamp)].data_rx1_re)
    rx1_im = np.array(radar_msg[int(timestamp)].data_rx1_im)
    rx2_re = np.array(radar_msg[int(timestamp)].data_rx2_re)
    rx2_im = np.array(radar_msg[int(timestamp)].data_rx2_im)
    # NOTE: I think the name of the variable is explainatory enough, no comment required

    # The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
    # and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).

    y = [rx1_re, rx1_im, rx2_re, rx2_im]  # Construct list with elemnts in order
    no_chirps = radar_msg[int(timestamp)].dimx  # 16 chirps
    length_chirp = radar_msg[int(timestamp)].dimy  # 128 elements in a chirp (each with rx 1 and rx 2 and Re and Im)
    chirps_temp = [[j[length_chirp * i:length_chirp * (i + 1)] for j in y] for i in range(no_chirps)]  # Chirps constructed
    chirps_temp = np.array(chirps_temp)

    final_list = [[], [], [], []]  # Average 16 chirps to 1 chirp for one timestamp 
    for i in range(128): 
        avg_calc_rx1re, avg_calc_rx1im, avg_calc_rx2re, avg_calc_rx2im = (np.array([]) for m in range(4))
        for k in range(16):
            # obtain one value from each chirp and appaned to respective list
            avg_calc_rx1re = np.append(avg_calc_rx1re, chirps_temp[k][0][i])
            avg_calc_rx1im = np.append(avg_calc_rx1im, chirps_temp[k][1][i])
            avg_calc_rx2re = np.append(avg_calc_rx2re, chirps_temp[k][2][i]) 
            avg_calc_rx2im = np.append(avg_calc_rx2im, chirps_temp[k][3][i]) 
        final_val_rx1re = np.average(avg_calc_rx1re)
        final_val_rx1im = np.average(avg_calc_rx1im)
        final_val_rx2re = np.average(avg_calc_rx2re) 
        final_val_rx2im = np.average(avg_calc_rx2im)
        
        final_val = [final_val_rx1re, final_val_rx1im, final_val_rx2re, final_val_rx2im]  # Remake list with only four averaged elemnets
        [final_list[l].append(o) for l, o in zip(range(4), final_val)]  # Append each value to specific list in final_list
    chirps = np.array(final_list)
    return chirps, no_chirps, length_chirp 
'''
def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j];
                  
        list1.remove(max1);
        final_list.append(max1)
    return final_list

def Nmaxelements(list1, N):
    final_list = []
  
    for i in range(0, N): 
        max1 = max(list1)
        final_list.append(max1)
        list1.remove(max1)
    return final_list '''