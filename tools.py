from scipy.signal import find_peaks # integrated function from scipy
import numpy as np
from math import pi, asin, sin
from matplotlib import pyplot as plt # plotting functions import
from matplotlib.widgets import Slider # function import for slider in window

def range_angle_calc(PSD, L, freq, chirp_time, phi_1, phi_2): #Calculate range
    
    PSD2 = PSD[L[0]:L[-1]] # called PSD2 and not PSD not to mess up with outside values 
    freq2 = freq[L[0]:L[-1]] # same reasoning as above 
    indices = find_peaks(PSD2)[0] # all indices corresponding to peaks
    # add treshold = X as an input to find_peaks function to implement a treshold

    freq_peaks = [freq2[indices[i]] for i in range(len(indices))] #initialize list 
    phase1_peaks = [phi_1[i] for i in range(len(indices))]  # Each peak found has the phase correponding to the frequency of a peak for Rx 1
    phase2_peaks = [phi_2[i] for i in range(len(indices))]  # For Rx 2
    
    # NOTE: Changed to list comprehension for conciseness (nice)
    ''' 
    for i in range(len(indices)): #go through all the peaks found
        index = indices[i] #index of the peak
        freq_peaks.append(freq2[index]) #move to list for rearrangement
        phase1_peaks.append(phi_1[i]) #for Rx 1
        phase2_peaks.append(phi_2[i]) #for Rx 2
    '''

    B = 250e6  # Hz (bandwidth range)
    c = 2.99792458e8  # m/s
    T = chirp_time * 16 # total time per frequency 
    
    d_test_2 = (sin((pi/180) * (76/2)))**-1 * (0.0125 / 2) #!!needs update!! Distance between two receivers
    f_temp = 24E9 # !!needs update!!
    range_lst = [] # list to colelct range data
    geo_angle_lst = [] # list to collect angle data
    velocity_lst = [] # list to store velocity data

    for k in range(len(freq_peaks)): # run through all the peaks detected
        try:
            range_lst = np.append(range_lst, c * T * freq_peaks[k] / (2 * B)) # range formula
            omega = phase2_peaks[k] - phase1_peaks[k]  # difference between phases 
            if abs(omega) > pi: # measure to ensure all angles make sense and are within phsyical real ranges
                omega = 2*pi - omega # convert angle 
            sine = c * omega / (2 * pi * f_temp * d_test_2) # angle formula 
            assert (-1 <= sine <= 1), "Can't calculate asin of value outside of <-1, 1>."
            geo_angle_lst = np.append(geo_angle_lst, (180 / pi)*asin(sine)) # final angel found in degrees
            velocity_lst = np.append(velocity_lst, (c * abs(omega) / (4 * pi * f_temp * T))) # final velocity found including velocity formula
        except Exception as err: # in the event that numbers are not calculable
            print("\033[;1m" + "Error: " + "\033[0;0m" + str(err))
            geo_angle_lst = np.append(geo_angle_lst, 0)
            velocity_lst = np.append(velocity_lst, 0)

    return range_lst, freq_peaks, geo_angle_lst # All data needed to be used in plots to determine obstacle parameters

def range_angle_velocity_calc2(freq1, freq2, phi_1, phi_2, chirp_time):
    B = 250e6  # Hz (bandwidth range)
    c = 2.99792458e8  # m/s
    T = chirp_time # total time
    
    d_test_2 = (sin((pi/180) * (76/2)))**-1 * (0.0125 / 2) # need update Distance between two receivers
    f_temp = 24E9 # need update
    range_lst1 = (c * T * freq1 / (2 * B) ) # range formula for receiver 1
    range_lst2 = (c * T * freq2 / (2 * B) ) # range formula for receiver 2
    
    delta_omega = phi_2 - phi_1  # difference between phases 
    temp_constant = c * delta_omega / (2 * pi * f_temp * d_test_2) # angle formula 
    
    geo_angle_lst = (180 / pi)*np.arcsin(temp_constant)  # final angle found in degrees
    velocity_lst = (c * abs(delta_omega) / (4 * pi * f_temp * T)) # final velocity found including velocity formula

    return range_lst1, range_lst2, geo_angle_lst, velocity_lst

# Either for reference, or requiring some change 
def phase_calc(FFT): # beautiful, outstanding, revolutionary
    return np.arctan(FFT.imag / FFT.real) # imaginary over real based on definitions
    # Might have to change to arctan2 if this does not work.



    



'''
def compute_angle_d(if1_i, if1_q, if2_i, if2_q, d_old, angle_offset_deg, wave_length_ant_spacing_ratio):

    # target_angle_data temp;
    # I believe it just means that the "temp" variable will be of custom "target_angle_data" data structure (Class)

    rx1_ang = get_phase(if1_i, if1_q) # - double(0.13 * PI);
    rx2_ang = get_phase(if2_i, if2_q)
    d_phi = (rx1_ang - rx2_ang)

    if (d_phi <= 0):
        d_phi += 2*PI

    d_phi -= PI

    if (int(d_old) == IGNORE_NAN):
        target_angle = 0
    else if ((d_phi > d_old + 0.9* PI) or (d_phi < d_old - 0.9* PI)):
       d_phi = d_old

    # Arcus sinus (-PI/2 to PI/2), input= -1..1
    target_angle = asin(d_phi * wave_length_ant_spacing_ratio / (2*PI))
    target_angle = target_angle * 180 / PI    # Angle (-90...90°)
    target_angle = target_angle + double(int(angle_offset_deg) + ANGLE_QUANTIZATION * 0.5)
    delta_angle  = fmodf(target_angle, double(ANGLE_QUANTIZATION)) # fmodf is floating-point remainder of the division operation x/y
    target_angle -= delta_angle
    temp.d_phi = d_phi
    temp.target_angle = target_angle

    return temp
'''

def fourier(chirps, t, realim, duration): #calculate the fourier of the signal 
    dt = duration / len(chirps[realim])  # realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t) # total number of timestamps

    f_hat = np.fft.fft(chirps[realim], n)  # frequency array already zero padded according to documentation

    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude  (np.real returns + 0*j)
    # Filtering:
    threshold = max(PSD)  # Find the highest peak
    indices = PSD > threshold / 1.5 # high-pass filter based on largest signal
    f_hat = f_hat * indices # Cleaned signal

    return f_hat

def PSD_calc(f_hat, t, duration, chirps): #intensty of frequenmcy calculated using fourier
    dt = duration / len(chirps[0])  # realim rx1 0,1 rx2 = 2,3  (0, 2 real; 1, 3 imaginary)
    n = len(t) # total number of timestamps
    
    PSD = np.real(f_hat * np.conj(f_hat) / n)  # Calculates the amplitude  (np.real returns + 0*j)

    freq = (1 / (dt * n)) * np.arange(n)  # frequency calculated
    L = np.arange(1, np.floor(n / 2), dtype='int') # prevent a divison of intensity over a twice long domain
    return PSD, freq, L 

def chirp_func(timestamp, radar_msg): # average the chirps for faster computation
    rx1_re = np.array(radar_msg[int(timestamp)].data_rx1_re) # obtain real part of rx 1
    rx1_im = np.array(radar_msg[int(timestamp)].data_rx1_im) # obtain imaginary part of rx 1
    rx2_re = np.array(radar_msg[int(timestamp)].data_rx2_re) # obtain real part of rx 2
    rx2_im = np.array(radar_msg[int(timestamp)].data_rx2_im) # obtain imaginary part of rx 2

    # The list 'chirps' is organised as follows. If the list is chirps[i][j] then i indicates the current chirp,
    # and j indicates the measurement type of that chirp (rx1_re or rx1_im etc.).

    y = [rx1_re, rx1_im, rx2_re, rx2_im] # construct list with elemnts in order
    no_chirps = radar_msg[int(timestamp)].dimx  # 16 chirps
    length_chirp = radar_msg[int(timestamp)].dimy # 128 elements in a chirp (each with rx 1 and rx 2 and Re and Im)
    chirps_temp = [[j[length_chirp * i:length_chirp * (i + 1)] for j in y] for i in range(no_chirps)] # chirps constructed
    chirps_temp = np.array(chirps_temp) # make to an array to use numpy

    final_list = [[], [], [], []]  # Average 16 chirps to 1 chirp for one timestamp 
    for i in range(128): 
        avg_calc_rx1re, avg_calc_rx1im, avg_calc_rx2re, avg_calc_rx2im = (np.array([]) for m in range(4)) # initialise arrays 
        for k in range(16):
            avg_calc_rx1re = np.append(avg_calc_rx1re, chirps_temp[k][0][i]) # obtain one value of rx1 re from each chirp and appaned to respective list
            avg_calc_rx1im = np.append(avg_calc_rx1im, chirps_temp[k][1][i]) # for rx 1 im
            avg_calc_rx2re = np.append(avg_calc_rx2re, chirps_temp[k][2][i]) # for rx 2 re
            avg_calc_rx2im = np.append(avg_calc_rx2im, chirps_temp[k][3][i]) # for rx 2 im
        final_val_rx1re = np.average(avg_calc_rx1re) # numpy average function for rx 1 re
        final_val_rx1im = np.average(avg_calc_rx1im) # same for rx 1 im
        final_val_rx2re = np.average(avg_calc_rx2re) # same for rx 2 re  
        final_val_rx2im = np.average(avg_calc_rx2im) # same for rx 2 im
        
        final_val = [final_val_rx1re, final_val_rx1im, final_val_rx2re, final_val_rx2im] # remake list with only four averaged elemnets
        [final_list[l].append(o) for l, o in zip(range(4), final_val)] # append each value to specific list in final_list
    chirps = np.array(final_list) # convert to array
    return chirps, no_chirps, length_chirp 